#include <metal_stdlib>
#include <render/shader_types.hpp>
#include <render/shadow_sampling.hpp>
using namespace metal;

constant uint FROXEL_SLICE_COUNT = 64u;

struct FroxelVertexOut {
    float4 clipPosition [[position]];
    float2 uv;
};

struct FroxelIntegratedOutput {
    float4 scatteringAndTransmittance [[color(0)]];
    float sceneDepth [[color(1)]];
};

static float2 clipUV(float2 uv) {
    return float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
}

static float3 reconstructWorldPosition(float2 uv, float depth, constant FroxelUniforms& uniforms) {
    const float4 worldH = uniforms.invViewProjection * float4(clipUV(uv), depth, 1.0f);
    return worldH.xyz / max(abs(worldH.w), 1.0e-6f) * sign(worldH.w);
}

static float3 viewRay(float2 uv, constant FroxelUniforms& uniforms) {
    const float3 farPosition = reconstructWorldPosition(uv, 1.0f, uniforms);
    return normalize(farPosition - uniforms.cameraPosition);
}

struct MediaHit {
    float3 worldPosition;
    float distance;
    float deviceDepth;
    float viewDepth;
    bool hasFiniteReceiver;
};

static MediaHit mediaHit(float2 uv, depth2d<float> sceneDepth, texture2d<float> cloudHitDepth,
                         constant FroxelUniforms& uniforms) {
    constexpr sampler pointSampler(mag_filter::nearest, min_filter::nearest,
                                   address::clamp_to_edge);
    constexpr sampler linearSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float opaqueDepth = sceneDepth.sample(pointSampler, uv);
    float3 position = reconstructWorldPosition(uv, opaqueDepth, uniforms);
    float distanceToHit = length(position - uniforms.cameraPosition);
    float deviceDepth = opaqueDepth;
    const float cloudDistance = cloudHitDepth.sample(linearSampler, uv).r;
    const bool hasFiniteReceiver = froxelHasFiniteReceiver(opaqueDepth, cloudDistance);
    if (cloudDistance > 0.0f && cloudDistance < distanceToHit) {
        position = uniforms.cameraPosition + viewRay(uv, uniforms) * cloudDistance;
        distanceToHit = cloudDistance;
        const float4 clip = uniforms.viewProjection * float4(position, 1.0f);
        deviceDepth = clip.z / max(abs(clip.w), 1.0e-6f) * sign(clip.w);
    }
    const float4 receiverClip = uniforms.viewProjection * float4(position, 1.0f);
    const float viewDepth = abs(receiverClip.w);
    return {position, distanceToHit, deviceDepth, viewDepth, hasFiniteReceiver};
}

static float henyeyGreenstein(float cosTheta, float anisotropy) {
    const float g = clamp(anisotropy, -0.90f, 0.90f);
    const float g2 = g * g;
    const float denominator = max(1.0f + g2 - 2.0f * g * cosTheta, 1.0e-4f);
    return (1.0f - g2) / (4.0f * 3.14159265359f * denominator * sqrt(denominator));
}

static float3 atmosphereRadiance(float3 direction, float3 lightDirection,
                                 texture2d<float> skyView) {
    constexpr sampler lutSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 viewHorizontal = direction.xz / max(length(direction.xz), 1.0e-5f);
    const float2 lightHorizontal = lightDirection.xz / max(length(lightDirection.xz), 1.0e-5f);
    const float crossValue =
        viewHorizontal.x * lightHorizontal.y - viewHorizontal.y * lightHorizontal.x;
    const float relativeAzimuth = atan2(crossValue, dot(viewHorizontal, lightHorizontal));
    const float2 uv = float2(relativeAzimuth / (2.0f * 3.14159265359f) + 0.5f,
                             saturate((direction.y + 0.08f) / 1.08f));
    return max(skyView.sample(lutSampler, uv).rgb, 0.0f);
}

static float4 regionalWeather(texture2d_array<float> weather, float2 cameraRelativeXZ,
                              constant WeatherMapUniforms& weatherMap) {
    constexpr sampler weatherSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 cell =
        (cameraRelativeXZ - weatherMap.originXZ) / max(weatherMap.cellSpacing, 1.0f);
    const float2 uv = (cell + 0.5f) / max(float2(weatherMap.gridSize), 1.0f);
    return mix(weather.sample(weatherSampler, uv, 0), weather.sample(weatherSampler, uv, 1),
               saturate(weatherMap.interpolation));
}

vertex FroxelVertexOut froxelFullscreenVertex(uint vertexID [[vertex_id]]) {
    const float2 positions[3] = {
        float2(-1.0f, -1.0f),
        float2(3.0f, -1.0f),
        float2(-1.0f, 3.0f),
    };
    FroxelVertexOut output;
    output.clipPosition = float4(positions[vertexID], 0.0f, 1.0f);
    output.uv = float2(positions[vertexID].x * 0.5f + 0.5f, 0.5f - positions[vertexID].y * 0.5f);
    return output;
}

// Stores local in-scattering source in RGB and extinction in A. Density is
// deterministic in world space, while illumination follows the same blended
// terrain shadows and cloud transmittance used by opaque surfaces.
kernel void froxelInjectKernel(
    texture3d<half, access::write> froxels [[texture(0)]],
    depth2d_array<float> nearShadow [[texture(1)]], depth2d_array<float> farShadow [[texture(2)]],
    depth2d<float> horizonShadow [[texture(3)]], texture2d<float> cloudShadow [[texture(4)]],
    texture2d<float> atmosphereSkyView [[texture(5)]],
    texture2d_array<float> weatherCloud [[texture(6)]],
    texture2d_array<float> weatherLayer [[texture(7)]], sampler shadowSampler [[sampler(1)]],
    constant FroxelUniforms& uniforms [[buffer(0)]], constant ShadowUniforms& shadow [[buffer(1)]],
    constant CloudShadowUniforms& cloud [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (any(gid >= uniforms.volumeDimensions.xyz)) {
        return;
    }

    const float nearDepth = max(uniforms.depthParams.x, 0.01f);
    const float farDepth = max(uniforms.depthParams.y, nearDepth + 0.01f);
    const float sliceNear =
        froxelSliceDepth(gid.z, uniforms.volumeDimensions.z, nearDepth, farDepth);
    const float sliceFar =
        froxelSliceDepth(gid.z + 1u, uniforms.volumeDimensions.z, nearDepth, farDepth);
    // A center-only injection turns the 160 by 104 screen grid into stable
    // shadow bands. Move the representative position through a deterministic
    // low-discrepancy sequence, then let the existing reprojection converge
    // the result. The offset stays inside this cell and logarithmic slice, so
    // it does not expand froxel coverage or alter an integrated segment length.
    const uint frameIndex = uniforms.volumeDimensions.w;
    const float spatialDither = interleavedGradientNoise(float2(gid.xy) + 0.5f);
    const float2 cellSequence = float2(froxelLowDiscrepancySample(frameIndex, 0u),
                                       froxelLowDiscrepancySample(frameIndex, 1u));
    const float depthSequence = froxelLowDiscrepancySample(frameIndex, 2u);
    const float2 cellOffset =
        fract(cellSequence + float2(spatialDither, fract(spatialDither * 1.61803399f))) - 0.5f;
    const float depthOffset = fract(depthSequence + fract(spatialDither * 2.41421356f)) - 0.5f;
    const float2 uv =
        (float2(gid.xy) + 0.5f + cellOffset * 0.80f) / float2(uniforms.volumeDimensions.xy);
    const float sliceSample = clamp(0.5f + depthOffset * 0.80f, 0.10f, 0.90f);
    const float sampleDepth = sliceNear * pow(sliceFar / sliceNear, sliceSample);
    const float3 rayDirection = viewRay(uv, uniforms);
    const float3 worldPosition = uniforms.cameraPosition + rayDirection * sampleDepth;

    const bool submergedCamera = uniforms.renderParams.w > 0.5f;
    const bool hasWaterPlane = uniforms.depthParams.w > -60000.0f;
    const bool belowWater = hasWaterPlane && worldPosition.y < uniforms.depthParams.w;
    if (submergedCamera || belowWater) {
        froxels.write(half4(0.0h), gid);
        return;
    }

    const float altitudeMeters =
        froxelAltitudeMeters(worldPosition.y, uniforms.physicalScale.y, uniforms.physicalScale.z);
    const float heightDensity = froxelHeightDensity(altitudeMeters, uniforms.mediumParams.w);
    float aerosol = max(uniforms.weatherParams.x, 0.0f);
    float humidity = saturate(uniforms.weatherParams.y);
    float precipitation = saturate(uniforms.weatherParams.z);
    if (uniforms.weatherMap.gridSize.x > 1u && uniforms.weatherMap.gridSize.y > 1u) {
        const float2 relativeXZ = worldPosition.xz - uniforms.cameraPosition.xz;
        const float4 cloudWeather = regionalWeather(weatherCloud, relativeXZ, uniforms.weatherMap);
        const float4 layerWeather = regionalWeather(weatherLayer, relativeXZ, uniforms.weatherMap);
        humidity = saturate(cloudWeather.y);
        precipitation = saturate(layerWeather.z);
        aerosol = max(layerWeather.w, 0.0f);
    }
    const bool hasRegionalWeather =
        uniforms.weatherMap.gridSize.x > 1u && uniforms.weatherMap.gridSize.y > 1u;
    const float weatherFog = hasRegionalWeather
                                 ? clamp(0.00006f + smoothstep(0.72f, 1.0f, humidity) * 0.0022f +
                                             precipitation * 0.0018f,
                                         0.00004f, 0.0060f)
                                 : max(uniforms.weatherParams.w, 0.0f);
    const float molecularExtinction = max(uniforms.mediumParams.x, 0.0f);
    const float extinction =
        max(molecularExtinction * heightDensity * (0.35f + aerosol) + weatherFog, 0.0f);

    if (extinction <= 1.0e-7f) {
        froxels.write(half4(0.0h), gid);
        return;
    }

    float terrainVisibility = 1.0f;
    if (uniforms.renderParams.z > 0.001f) {
        terrainVisibility = sampleShadowVisibilityFast(worldPosition, 1.0f, nearShadow, farShadow,
                                                       horizonShadow, shadowSampler, shadow);
    }

    float cloudTransmittance = 1.0f;
    const float cloudFootprint = max(cloud.footprintAndTexel.x, 1.0f);
    const float cloudTexel = max(cloud.footprintAndTexel.y, 1.0f);
    const float2 cloudCenter = floor(cloud.cameraPosition.xz / cloudTexel) * cloudTexel;
    const float2 referencePosition =
        cloudShadowReferencePosition(worldPosition, normalize(cloud.sunDirection));
    const float2 cloudUV = (referencePosition - cloudCenter) / cloudFootprint + 0.5f;
    if (all(cloudUV >= 0.0f) && all(cloudUV <= 1.0f)) {
        constexpr sampler cloudSampler(mag_filter::linear, min_filter::linear,
                                       address::clamp_to_edge);
        cloudTransmittance = mix(1.0f, cloudShadow.sample(cloudSampler, cloudUV).r,
                                 saturate(cloud.footprintAndTexel.z));
    }

    const float scatteringAlbedo = saturate(uniforms.mediumParams.y);
    const float sigmaS = extinction * scatteringAlbedo;
    const float phase = henyeyGreenstein(dot(rayDirection, normalize(uniforms.lightDirection)),
                                         uniforms.mediumParams.z);
    const float directVisibility = terrainVisibility * cloudTransmittance;
    const float directScattering = phase * 12.5663706f * directVisibility;
    const float3 molecularColor =
        normalize(float3(0.58f, 0.78f, 1.0f) + aerosol * float3(0.42f, 0.22f, 0.04f) +
                  precipitation * float3(0.18f, 0.20f, 0.22f));
    const float3 skyRadiance =
        atmosphereRadiance(rayDirection, uniforms.solarDirection, atmosphereSkyView);
    const float3 directSource = uniforms.lightRadiance * directScattering;
    const float3 ambientSource = skyRadiance * (0.18f + 0.35f * humidity);
    const float3 source = sigmaS * molecularColor * (directSource + ambientSource);
    froxels.write(half4(half3(max(source, 0.0f)), half(extinction)), gid);
}

// Integrates each low-resolution froxel column once. Half-resolution pixels
// then resolve this cumulative volume at their opaque hit depth, avoiding a
// 64-step march for every display pixel.
kernel void froxelIntegrateKernel(texture3d<float, access::read> froxels [[texture(0)]],
                                  texture3d<half, access::write> integrated [[texture(1)]],
                                  constant FroxelUniforms& uniforms [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= uniforms.volumeDimensions.xy)) {
        return;
    }
    const float nearDepth = max(uniforms.depthParams.x, 0.01f);
    const float farDepth = max(uniforms.depthParams.y, nearDepth + 0.01f);
    float3 scattering = 0.0f;
    float transmittance = 1.0f;
    for (uint slice = 0u; slice < FROXEL_SLICE_COUNT; ++slice) {
        if (slice >= uniforms.volumeDimensions.z) {
            break;
        }
        const float sliceNear =
            froxelSliceDepth(slice, uniforms.volumeDimensions.z, nearDepth, farDepth);
        const float sliceFar =
            froxelSliceDepth(slice + 1u, uniforms.volumeDimensions.z, nearDepth, farDepth);
        const float segmentLength =
            froxelPhysicalDistance(sliceFar - sliceNear, uniforms.physicalScale.x);
        const uint3 coordinate = uint3(gid, slice);
        const float4 medium = froxels.read(coordinate);
        const float segmentTransmittance = beerLambertTransmittance(medium.a, segmentLength);
        const float3 segmentScattering =
            medium.a > 1.0e-6f ? medium.rgb * ((1.0f - segmentTransmittance) / medium.a)
                               : medium.rgb * segmentLength;
        scattering += transmittance * segmentScattering;
        transmittance *= segmentTransmittance;
        integrated.write(half4(half3(max(scattering, 0.0f)), half(saturate(transmittance))),
                         coordinate);
    }
}

fragment FroxelIntegratedOutput froxelResolveFragment(FroxelVertexOut in [[stage_in]],
                                                      depth2d<float> sceneDepth [[texture(0)]],
                                                      texture3d<float> integrated [[texture(1)]],
                                                      texture2d<float> cloudHitDepth [[texture(2)]],
                                                      constant FroxelUniforms& uniforms
                                                      [[buffer(0)]]) {
    constexpr sampler linearSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const MediaHit hit = mediaHit(in.uv, sceneDepth, cloudHitDepth, uniforms);
    if (!hit.hasFiniteReceiver) {
        FroxelIntegratedOutput output;
        output.scatteringAndTransmittance = float4(0.0f, 0.0f, 0.0f, 1.0f);
        output.sceneDepth = 1.0f;
        return output;
    }
    const float sceneDistance = hit.distance;
    const float nearDepth = max(uniforms.depthParams.x, 0.01f);
    const float farDepth = max(uniforms.depthParams.y, nearDepth + 0.01f);
    const float normalizedDepth =
        saturate(log(max(sceneDistance, nearDepth) / nearDepth) / log(farDepth / nearDepth));
    const float sliceCoordinate =
        clamp(normalizedDepth - 0.5f / float(uniforms.volumeDimensions.z), 0.0f, 1.0f);
    float4 medium = integrated.sample(linearSampler, float3(in.uv, sliceCoordinate));
    const float firstSliceWeight = saturate(normalizedDepth * float(uniforms.volumeDimensions.z));
    medium = mix(float4(0.0f, 0.0f, 0.0f, 1.0f), medium, firstSliceWeight);
    FroxelIntegratedOutput output;
    output.scatteringAndTransmittance = float4(max(medium.rgb, 0.0f), saturate(medium.a));
    // Preserve linear view depth in R32Float. Device depth is highly
    // compressed at grazing cave floors and would reject valid temporal
    // history before the dithered froxel samples can converge.
    output.sceneDepth = hit.viewDepth;
    return output;
}

static float4 currentNeighborhoodClamp(texture2d<float> current, uint2 pixel, float4 history) {
    const int2 extent = int2(current.get_width(), current.get_height());
    float4 neighborhoodMin = float4(INFINITY);
    float4 neighborhoodMax = float4(-INFINITY);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const uint2 samplePixel = uint2(clamp(int2(pixel) + int2(x, y), int2(0), extent - 1));
            const float4 value = current.read(samplePixel);
            neighborhoodMin = min(neighborhoodMin, value);
            neighborhoodMax = max(neighborhoodMax, value);
        }
    }
    return clamp(history, neighborhoodMin, neighborhoodMax);
}

fragment FroxelIntegratedOutput froxelReprojectFragment(
    FroxelVertexOut in [[stage_in]], texture2d<float> current [[texture(0)]],
    texture2d<float> currentDepth [[texture(1)]], texture2d<float> history [[texture(2)]],
    texture2d<float> historyDepth [[texture(3)]], depth2d<float> sceneDepth [[texture(4)]],
    texture2d<float> cloudHitDepth [[texture(5)]],
    constant FroxelUniforms& uniforms [[buffer(0)]]) {
    constexpr sampler linearSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const uint2 pixel = uint2(in.clipPosition.xy);
    const float4 currentValue = current.read(pixel);
    const float currentViewDepth = currentDepth.read(pixel).r;
    const MediaHit currentHit = mediaHit(in.uv, sceneDepth, cloudHitDepth, uniforms);
    const float3 worldPosition = currentHit.worldPosition;
    const float4 previousClip = uniforms.previousViewProjection * float4(worldPosition, 1.0f);
    const float3 previousNdc =
        previousClip.xyz / max(abs(previousClip.w), 1.0e-6f) * sign(previousClip.w);
    const float2 previousUV = float2(previousNdc.x * 0.5f + 0.5f, 0.5f - previousNdc.y * 0.5f);
    const bool inside = previousClip.w > 0.0f && all(previousUV >= 0.0f) &&
                        all(previousUV <= 1.0f) && previousNdc.z >= 0.0f && previousNdc.z <= 1.0f;
    const float expectedPreviousViewDepth = abs(previousClip.w);
    const float depthThreshold = froxelTemporalLinearDepthTolerance(expectedPreviousViewDepth);
    bool valid = uniforms.renderParams.y > 0.5f && currentHit.hasFiniteReceiver && inside &&
                 isfinite(currentViewDepth) && currentViewDepth > 0.0f;
    float depthError = INFINITY;
    if (valid) {
        const float previousViewDepth = historyDepth.sample(linearSampler, previousUV).r;
        depthError = abs(previousViewDepth - expectedPreviousViewDepth);
        valid = isfinite(previousViewDepth) && isfinite(expectedPreviousViewDepth) &&
                depthThreshold > 0.0f && depthError <= depthThreshold;
    }

    float4 filtered = currentValue;
    if (valid) {
        float4 previousValue = history.sample(linearSampler, previousUV);
        previousValue = currentNeighborhoodClamp(current, pixel, previousValue);
        const float confidence = saturate(1.0f - depthError / depthThreshold);
        const float historyWeight = saturate(uniforms.renderParams.x) * confidence;
        filtered = mix(currentValue, previousValue, historyWeight);
    }

    FroxelIntegratedOutput output;
    output.scatteringAndTransmittance = float4(max(filtered.rgb, 0.0f), saturate(filtered.a));
    output.sceneDepth = currentViewDepth;
    return output;
}

static float4 bilateralMediumSample(texture2d<float> integrated, texture2d<float> integratedDepth,
                                    float2 uv, float fullDepth) {
    const float2 size = float2(integrated.get_width(), integrated.get_height());
    const float2 texelPosition = uv * size - 0.5f;
    const int2 base = int2(floor(texelPosition));
    const float2 fraction = fract(texelPosition);
    const int2 extent = int2(integrated.get_width(), integrated.get_height());
    float4 result = 0.0f;
    float totalWeight = 0.0f;
    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            const int2 offset = int2(x, y);
            const uint2 samplePixel = uint2(clamp(base + offset, int2(0), extent - 1));
            const float spatialWeight = (x == 0 ? 1.0f - fraction.x : fraction.x) *
                                        (y == 0 ? 1.0f - fraction.y : fraction.y);
            const float halfDepth = integratedDepth.read(samplePixel).r;
            const float depthWeight = froxelBilateralLinearDepthWeight(fullDepth, halfDepth);
            const float weight = spatialWeight * depthWeight;
            result += integrated.read(samplePixel) * weight;
            totalWeight += weight;
        }
    }
    if (totalWeight < 1.0e-5f) {
        const uint2 nearest = uint2(clamp(int2(uv * size), int2(0), extent - 1));
        return integrated.read(nearest);
    }
    return result / totalWeight;
}

// Fixed-function blending is configured as source + destination * source.a,
// giving the required scattering + scene * transmittance composite in one
// bandwidth-efficient pass.
fragment float4 froxelCompositeFragment(FroxelVertexOut in [[stage_in]],
                                        texture2d<float> integrated [[texture(0)]],
                                        texture2d<float> integratedDepth [[texture(1)]],
                                        depth2d<float> sceneDepth [[texture(2)]],
                                        texture2d<float> cloudHitDepth [[texture(3)]],
                                        constant FroxelUniforms& uniforms [[buffer(0)]]) {
    if (uniforms.renderParams.w > 0.5f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    const MediaHit hit = mediaHit(in.uv, sceneDepth, cloudHitDepth, uniforms);
    if (!hit.hasFiniteReceiver) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    const float4 medium = bilateralMediumSample(integrated, integratedDepth, in.uv, hit.viewDepth);
    return float4(max(medium.rgb, 0.0f), saturate(medium.a));
}

// Disabled froxels retain view-independent atmosphere aerial perspective at
// one depth sample per pixel, while omitting shadowed shafts and temporal
// storage. Underwater returns identity because water owns that medium.
fragment float4 aerialPerspectiveFragment(FroxelVertexOut in [[stage_in]],
                                          depth2d<float> sceneDepth [[texture(0)]],
                                          texture2d<float> atmosphereSkyView [[texture(1)]],
                                          texture2d<float> cloudHitDepth [[texture(2)]],
                                          constant FroxelUniforms& uniforms [[buffer(0)]]) {
    if (uniforms.renderParams.w > 0.5f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    const MediaHit hit = mediaHit(in.uv, sceneDepth, cloudHitDepth, uniforms);
    if (!hit.hasFiniteReceiver) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    const float3 scenePosition = hit.worldPosition;
    const float farDepth = max(uniforms.depthParams.y, 1.0f);
    const float distance =
        froxelPhysicalDistance(min(hit.distance, farDepth), uniforms.physicalScale.x);
    const float aerosol = max(uniforms.weatherParams.x, 0.0f);
    const float humidity = saturate(uniforms.weatherParams.y);
    const float precipitation = saturate(uniforms.weatherParams.z);
    const float extinction = max(uniforms.mediumParams.x, 0.0f) * (0.35f + aerosol) +
                             max(uniforms.weatherParams.w, 0.0f);
    const float transmittance = beerLambertTransmittance(extinction, distance);
    const float3 rayDirection = normalize(scenePosition - uniforms.cameraPosition);
    const float3 skyRadiance =
        atmosphereRadiance(rayDirection, uniforms.solarDirection, atmosphereSkyView);
    const float3 atmosphereColor = mix(skyRadiance, max(uniforms.lightRadiance, 0.0f) * 0.08f,
                                       saturate(humidity * 0.35f + precipitation * 0.15f));
    const float3 scattering = atmosphereColor * (1.0f - transmittance);
    return float4(scattering, transmittance);
}

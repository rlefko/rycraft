#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

struct FullscreenVertex {
    float4 position [[position]];
    float2 uv;
};

namespace {

FullscreenVertex fullscreenTriangle(uint vertexID) {
    const float2 positions[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    FullscreenVertex result;
    result.position = float4(positions[vertexID], 0.0f, 1.0f);
    result.uv = float2(positions[vertexID].x * 0.5f + 0.5f, 0.5f - positions[vertexID].y * 0.5f);
    return result;
}

float4 weatherField(texture2d_array<float> weather, float2 worldXZ,
                    constant WeatherMapUniforms& weatherMap) {
    constexpr sampler weatherSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 uv = weatherMapTextureCoordinate(worldXZ, weatherMap);
    return mix(weather.sample(weatherSampler, uv, 0), weather.sample(weatherSampler, uv, 1),
               weatherMap.interpolation);
}

float2 decodeCloudMotion(float4 encoded, float wrapBlocks) {
    float xAngle = atan2(encoded.y, encoded.x);
    float zAngle = atan2(encoded.w, encoded.z);
    if (xAngle < 0.0f) xAngle += 2.0f * M_PI_F;
    if (zAngle < 0.0f) zAngle += 2.0f * M_PI_F;
    return float2(xAngle, zAngle) * (wrapBlocks / (2.0f * M_PI_F));
}

float2 weatherMotionField(texture2d_array<float> weatherMotion, float2 worldXZ,
                          constant WeatherMapUniforms& weatherMap, uint firstSlice) {
    constexpr sampler weatherSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 uv = weatherMapTextureCoordinate(worldXZ, weatherMap);
    const float4 encoded =
        mix(weatherMotion.sample(weatherSampler, uv, firstSlice),
            weatherMotion.sample(weatherSampler, uv, firstSlice + 1), weatherMap.interpolation);
    return decodeCloudMotion(encoded, weatherMap.motionWrapBlocks);
}

float2 wrappedMotionDelta(float2 current, float2 previous, float wrapBlocks) {
    float2 delta = current - previous;
    return delta - round(delta / wrapBlocks) * wrapBlocks;
}

float4 cloudProfileWeights(float type) {
    const float scaledType = saturate(type) * 4.0f;
    return saturate(1.0f - abs(scaledType - float4(1.0f, 2.0f, 3.0f, 4.0f)));
}

float cloudHeightProfile(float normalizedHeight, float type) {
    normalizedHeight = saturate(normalizedHeight);
    const float stratus = smoothstep(0.0f, 0.12f, normalizedHeight) *
                          (1.0f - smoothstep(0.72f, 1.0f, normalizedHeight));
    const float cumulus = smoothstep(0.0f, 0.18f, normalizedHeight) *
                          (1.0f - smoothstep(0.72f, 1.0f, normalizedHeight)) *
                          mix(0.55f, 1.0f, smoothstep(0.12f, 0.45f, normalizedHeight));
    const float cumulonimbus = smoothstep(0.0f, 0.08f, normalizedHeight) *
                               (1.0f - smoothstep(0.92f, 1.0f, normalizedHeight));
    const float cirrus = smoothstep(0.05f, 0.32f, normalizedHeight) *
                         (1.0f - smoothstep(0.62f, 0.98f, normalizedHeight));
    const float4 weights = cloudProfileWeights(type);
    return dot(weights, float4(cirrus, stratus, cumulus, cumulonimbus));
}

float cloudHighMotionWeight(float normalizedHeight, float type) {
    const float4 weights = cloudProfileWeights(type);
    const float stormTop = weights.z * smoothstep(0.48f, 0.90f, normalizedHeight) * 0.72f;
    const float cumulusTop = weights.y * smoothstep(0.68f, 0.96f, normalizedHeight) * 0.18f;
    return saturate(weights.w + stormTop + cumulusTop);
}

float2 cloudLayerMotion(texture2d_array<float> weatherMotion, float2 worldXZ,
                        constant WeatherMapUniforms& weatherMap, float highWeight) {
    const float2 low = weatherMotionField(weatherMotion, worldXZ, weatherMap, 0);
    const float2 high = weatherMotionField(weatherMotion, worldXZ, weatherMap, 2);
    return low + wrappedMotionDelta(high, low, weatherMap.motionWrapBlocks) * highWeight;
}

float3 cloudNoiseCoordinate(float3 worldPosition, float2 motion, texture2d<float> curlNoise) {
    constexpr sampler repeatLinear(coord::normalized, address::repeat, filter::linear);
    const float2 curl = curlNoise.sample(repeatLinear, worldPosition.xz * 0.0007f).rg;
    return float3((worldPosition.xz - motion + curl * 48.0f) * CLOUD_NOISE_BLOCK_FREQUENCY,
                  worldPosition.y * 0.0032f)
        .xzy;
}

float cloudNoiseDensity(float3 coordinate, float normalizedHeight, float4 cloud,
                        texture3d<float> baseNoise, texture3d<float> erosionNoise,
                        float erosionStrength, float densityScale) {
    constexpr sampler repeatLinear(coord::normalized, address::repeat, filter::linear);
    const float base = baseNoise.sample(repeatLinear, coordinate).r;
    const float erosion =
        erosionNoise
            .sample(repeatLinear, coordinate * 4.0f + float3(0.0f, normalizedHeight * 0.12f, 0.0f))
            .r;
    const float threshold = mix(0.78f, 0.30f, cloud.x);
    float density = saturate((base - threshold) / max(1.0f - threshold, 0.05f));
    density = saturate(density - (1.0f - erosion) * erosionStrength);
    density *= cloudHeightProfile(normalizedHeight, cloud.w);
    density *= mix(0.55f, 1.25f, cloud.y) * densityScale;
    return saturate(density);
}

float sampleCloudDensity(float3 worldPosition, texture3d<float> baseNoise,
                         texture3d<float> erosionNoise, texture2d<float> curlNoise,
                         texture2d_array<float> weatherCloud, texture2d_array<float> weatherLayer,
                         texture2d_array<float> weatherMotion,
                         constant CloudRenderUniforms& uniforms) {
    const float2 relativeXZ = worldPosition.xz - uniforms.cameraPosition.xz;
    const float4 cloud = weatherField(weatherCloud, relativeXZ, uniforms.weatherMap);
    const float4 layer = weatherField(weatherLayer, relativeXZ, uniforms.weatherMap);
    if (cloud.x < 0.015f || layer.y <= layer.x + 1.0f) {
        return 0.0f;
    }
    const float normalizedHeight = (worldPosition.y - layer.x) / (layer.y - layer.x);
    if (normalizedHeight <= 0.0f || normalizedHeight >= 1.0f) {
        return 0.0f;
    }
    const float2 motion = cloudLayerMotion(weatherMotion, relativeXZ, uniforms.weatherMap,
                                           cloudHighMotionWeight(normalizedHeight, cloud.w));
    const float3 coordinate = cloudNoiseCoordinate(worldPosition, motion, curlNoise);
    return cloudNoiseDensity(coordinate, normalizedHeight, cloud, baseNoise, erosionNoise,
                             uniforms.densityParams.y, uniforms.densityParams.x);
}

float sampleCloudDensityShadow(float3 worldPosition, texture3d<float> baseNoise,
                               texture3d<float> erosionNoise, texture2d<float> curlNoise,
                               texture2d_array<float> weatherCloud,
                               texture2d_array<float> weatherLayer,
                               texture2d_array<float> weatherMotion,
                               constant CloudShadowUniforms& uniforms) {
    const float2 relativeXZ = worldPosition.xz - uniforms.cameraPosition.xz;
    const float4 cloud = weatherField(weatherCloud, relativeXZ, uniforms.weatherMap);
    const float4 layer = weatherField(weatherLayer, relativeXZ, uniforms.weatherMap);
    if (cloud.x < 0.015f || layer.y <= layer.x + 1.0f) {
        return 0.0f;
    }
    const float normalizedHeight = (worldPosition.y - layer.x) / (layer.y - layer.x);
    if (normalizedHeight <= 0.0f || normalizedHeight >= 1.0f) {
        return 0.0f;
    }
    const float2 motion = cloudLayerMotion(weatherMotion, relativeXZ, uniforms.weatherMap,
                                           cloudHighMotionWeight(normalizedHeight, cloud.w));
    const float3 coordinate = cloudNoiseCoordinate(worldPosition, motion, curlNoise);
    return cloudNoiseDensity(coordinate, normalizedHeight, cloud, baseNoise, erosionNoise, 0.38f,
                             1.0f);
}

float dualLobePhase(float cosine, float forwardG, float backwardG, float blend) {
    const float forward =
        (1.0f - forwardG * forwardG) /
        (4.0f * M_PI_F *
         pow(max(1.0f + forwardG * forwardG - 2.0f * forwardG * cosine, 1.0e-4f), 1.5f));
    const float backward =
        (1.0f - backwardG * backwardG) /
        (4.0f * M_PI_F *
         pow(max(1.0f + backwardG * backwardG - 2.0f * backwardG * cosine, 1.0e-4f), 1.5f));
    return mix(backward, forward, blend);
}

float3 worldRay(float2 uv, constant CloudRenderUniforms& uniforms) {
    const float4 farClip = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 1.0f, 1.0f);
    const float4 farWorld = uniforms.invViewProjection * farClip;
    return normalize(farWorld.xyz / max(farWorld.w, 1.0e-6f) - uniforms.cameraPosition);
}

float sceneDistance(float2 uv, depth2d<float> sceneDepth, constant CloudRenderUniforms& uniforms) {
    constexpr sampler pointSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    const float depth = sceneDepth.sample(pointSampler, uv);
    if (depth >= 1.0f) {
        return 1.0e7f;
    }
    const float4 clip = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f);
    const float4 world = uniforms.invViewProjection * clip;
    return length(world.xyz / max(world.w, 1.0e-6f) - uniforms.cameraPosition);
}

} // namespace

kernel void volumetricCloudMarchKernel(
    texture3d<float> baseNoise [[texture(0)]], texture3d<float> erosionNoise [[texture(1)]],
    texture2d<float> curlNoise [[texture(2)]], texture2d_array<float> weatherCloud [[texture(3)]],
    texture2d_array<float> weatherLayer [[texture(4)]],
    texture2d_array<float> weatherMotion [[texture(5)]], depth2d<float> sceneDepth [[texture(6)]],
    texture2d<float, access::write> output [[texture(7)]],
    texture2d<float, access::write> hitDepthOutput [[texture(8)]],
    constant CloudRenderUniforms& uniforms [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    const float2 uv = (float2(gid) + 0.5f) / float2(output.get_width(), output.get_height());
    const float3 ray = worldRay(uv, uniforms);
    const float lowY = uniforms.layerBounds.x;
    const float highY = uniforms.layerBounds.y;
    const float maximumDistance = cloudMarchRayDistanceLimit(
        ray, uniforms.cameraForward, uniforms.layerBounds.w, uniforms.weatherMap);
    float entry = 0.0f;
    float exit = maximumDistance;
    if (abs(ray.y) < 1.0e-5f) {
        if (uniforms.cameraPosition.y < lowY || uniforms.cameraPosition.y > highY) {
            output.write(0.0f, gid);
            hitDepthOutput.write(0.0f, gid);
            return;
        }
    } else {
        const float t0 = (lowY - uniforms.cameraPosition.y) / ray.y;
        const float t1 = (highY - uniforms.cameraPosition.y) / ray.y;
        entry = max(min(t0, t1), 0.0f);
        exit = min(max(t0, t1), maximumDistance);
    }
    exit = min(exit, sceneDistance(uv, sceneDepth, uniforms));
    if (exit <= entry) {
        output.write(0.0f, gid);
        hitDepthOutput.write(0.0f, gid);
        return;
    }

    uint viewSteps = uint(uniforms.renderParams.x + 0.5f);
    const uint lightSteps = uint(uniforms.renderParams.y + 0.5f);
    const float horizontalSpan = (exit - entry) * length(ray.xz);
    const uint weatherCoverageSteps =
        uint(ceil(horizontalSpan / max(uniforms.weatherMap.cellSpacing, 1.0f)));
    viewSteps = max(viewSteps, min(weatherCoverageSteps, 64u));
    const float stepLength = (exit - entry) / float(viewSteps);
    const float jitter = fract(sin(dot(float2(gid), float2(12.9898f, 78.233f)) +
                                   uniforms.resolutionAndFrame.z * 0.6180339f) *
                               43758.5453f);
    float transmittance = 1.0f;
    float3 scattering = 0.0f;
    float hitDepth = 0.0f;
    const float phase = dualLobePhase(dot(ray, uniforms.sunDirection), uniforms.phaseParams.x,
                                      uniforms.phaseParams.y, uniforms.phaseParams.z);
    for (uint step = 0; step < viewSteps; ++step) {
        const float segmentBegin = entry + float(step) * stepLength;
        const float segmentEnd = min(segmentBegin + stepLength, exit);
        const float segmentMiddle = (segmentBegin + segmentEnd) * 0.5f;
        const float3 middlePosition = uniforms.cameraPosition + ray * segmentMiddle;
        const float2 middleRelativeXZ = middlePosition.xz - uniforms.cameraPosition.xz;
        const float4 localLayer = weatherField(weatherLayer, middleRelativeXZ, uniforms.weatherMap);
        const float2 localInterval = cloudRaySegmentLayerIntersection(
            uniforms.cameraPosition.y, ray.y, segmentBegin, segmentEnd, localLayer.x, localLayer.y);
        if (localInterval.y <= localInterval.x) {
            continue;
        }
        const float localJitter = fract(jitter + float(step) * 0.6180339f);
        const float distance = mix(localInterval.x, localInterval.y, localJitter);
        const float localStepLength = localInterval.y - localInterval.x;
        const float3 position = uniforms.cameraPosition + ray * distance;
        const float density =
            sampleCloudDensity(position, baseNoise, erosionNoise, curlNoise, weatherCloud,
                               weatherLayer, weatherMotion, uniforms);
        if (density > 0.001f) {
            if (hitDepth == 0.0f) {
                hitDepth = distance;
            }
            float lightTransmittance = 1.0f;
            const float lightStepLength = max(uniforms.layerBounds.z, 12.0f);
            float3 lightPosition = position;
            for (uint lightStep = 0; lightStep < lightSteps; ++lightStep) {
                lightPosition += uniforms.sunDirection * lightStepLength;
                const float lightDensity =
                    sampleCloudDensity(lightPosition, baseNoise, erosionNoise, curlNoise,
                                       weatherCloud, weatherLayer, weatherMotion, uniforms);
                lightTransmittance *= beerLambertTransmittance(
                    lightDensity * uniforms.densityParams.z, lightStepLength);
            }
            const float extinction = density * uniforms.densityParams.z;
            const float sampleTransmittance = beerLambertTransmittance(extinction, localStepLength);
            const float3 skyIrradiance =
                uniforms.sunRadiance * lightTransmittance * phase +
                uniforms.skyIrradiance *
                    (0.55f + uniforms.densityParams.w * (1.0f - lightTransmittance));
            const float3 groundIrradiance =
                uniforms.skyIrradiance * float3(0.24f, 0.27f, 0.21f) *
                smoothstep(0.0f, 0.35f,
                           (position.y - localLayer.x) / max(localLayer.y - localLayer.x, 1.0f));
            scattering +=
                transmittance * (skyIrradiance + groundIrradiance) * (1.0f - sampleTransmittance);
            transmittance *= sampleTransmittance;
            if (transmittance < 0.01f) {
                break;
            }
        }
    }
    output.write(float4(scattering, 1.0f - transmittance), gid);
    hitDepthOutput.write(hitDepth, gid);
}

kernel void
volumetricCloudTemporalKernel(texture2d<float, access::read> current [[texture(0)]],
                              texture2d<float, access::read> currentDepth [[texture(1)]],
                              texture2d<float, access::sample> history [[texture(2)]],
                              texture2d<float, access::sample> historyDepth [[texture(3)]],
                              texture2d<float, access::write> output [[texture(4)]],
                              texture2d<float, access::write> outputDepth [[texture(5)]],
                              texture2d_array<float> currentWeatherMotion [[texture(6)]],
                              texture2d_array<float> previousWeatherMotion [[texture(7)]],
                              texture2d_array<float> currentWeatherCloud [[texture(8)]],
                              texture2d_array<float> currentWeatherLayer [[texture(9)]],
                              texture2d_array<float> previousWeatherCloud [[texture(10)]],
                              texture2d_array<float> previousWeatherLayer [[texture(11)]],
                              constant CloudRenderUniforms& uniforms [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 size = float2(output.get_width(), output.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;
    const float4 value = current.read(gid);
    const float depth = currentDepth.read(gid).r;
    float4 minimumValue = value;
    float4 maximumValue = value;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const int2 coordinate = clamp(int2(gid) + int2(x, y), int2(0),
                                          int2(output.get_width() - 1, output.get_height() - 1));
            const float4 neighbor = current.read(uint2(coordinate));
            minimumValue = min(minimumValue, neighbor);
            maximumValue = max(maximumValue, neighbor);
        }
    }
    const float3 hit = uniforms.cameraPosition + worldRay(uv, uniforms) * depth;
    const float2 relativeXZ = hit.xz - uniforms.cameraPosition.xz;
    const float4 currentCloud = weatherField(currentWeatherCloud, relativeXZ, uniforms.weatherMap);
    const float4 currentLayer = weatherField(currentWeatherLayer, relativeXZ, uniforms.weatherMap);
    const float4 previousCloud =
        weatherField(previousWeatherCloud, relativeXZ, uniforms.previousWeatherMap);
    const float4 previousLayer =
        weatherField(previousWeatherLayer, relativeXZ, uniforms.previousWeatherMap);
    const float currentHeight =
        saturate((hit.y - currentLayer.x) / max(currentLayer.y - currentLayer.x, 1.0f));
    const float previousHeight =
        saturate((hit.y - previousLayer.x) / max(previousLayer.y - previousLayer.x, 1.0f));
    const float2 currentMotion =
        cloudLayerMotion(currentWeatherMotion, relativeXZ, uniforms.weatherMap,
                         cloudHighMotionWeight(currentHeight, currentCloud.w));
    const float2 previousMotion =
        cloudLayerMotion(previousWeatherMotion, relativeXZ, uniforms.previousWeatherMap,
                         cloudHighMotionWeight(previousHeight, previousCloud.w));
    const float2 motionDelta =
        wrappedMotionDelta(currentMotion, previousMotion, uniforms.weatherMap.motionWrapBlocks);
    const float3 previousHit = hit - float3(motionDelta.x, 0.0f, motionDelta.y);
    const float4 previousClip = uniforms.previousViewProjection * float4(previousHit, 1.0f);
    const float2 previousUv =
        previousClip.xy / max(previousClip.w, 1.0e-6f) * float2(0.5f, -0.5f) + 0.5f;
    const bool inside = all(previousUv >= 0.0f) && all(previousUv <= 1.0f);
    bool valid = uniforms.renderParams.z > 0.5f && inside && depth > 0.0f;
    if (valid) {
        const float oldDepth = historyDepth.sample(linearSampler, previousUv).r;
        valid = abs(oldDepth - depth) < max(depth * 0.12f, 16.0f);
    }
    float4 oldValue = value;
    if (valid) {
        oldValue = clamp(history.sample(linearSampler, previousUv), minimumValue, maximumValue);
    }
    output.write(mix(value, oldValue, valid ? 0.88f : 0.0f), gid);
    outputDepth.write(depth, gid);
}

kernel void cloudShadowKernel(texture3d<float> baseNoise [[texture(0)]],
                              texture3d<float> erosionNoise [[texture(1)]],
                              texture2d<float> curlNoise [[texture(2)]],
                              texture2d_array<float> weatherCloud [[texture(3)]],
                              texture2d_array<float> weatherLayer [[texture(4)]],
                              texture2d_array<float> weatherMotion [[texture(5)]],
                              texture2d<float, access::write> output [[texture(6)]],
                              constant CloudShadowUniforms& uniforms [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    const float2 uv = (float2(gid) + 0.5f) / float2(output.get_width(), output.get_height());
    const float2 center = floor(uniforms.cameraPosition.xz / uniforms.footprintAndTexel.y) *
                          uniforms.footprintAndTexel.y;
    const float2 worldXZ = center + (uv - 0.5f) * uniforms.footprintAndTexel.x;
    const float4 layer =
        weatherField(weatherLayer, worldXZ - uniforms.cameraPosition.xz, uniforms.weatherMap);
    const float3 direction = normalize(uniforms.sunDirection);
    if (direction.y <= 0.03f) {
        output.write(1.0f, gid);
        return;
    }
    constexpr float RECEIVER_HEIGHT = 64.0f;
    const float entryDistance = max(layer.x - RECEIVER_HEIGHT, 0.0f) / direction.y;
    float3 position = float3(worldXZ.x, RECEIVER_HEIGHT, worldXZ.y) + direction * entryDistance;
    const uint stepCount = uniforms.footprintAndTexel.z > 1.5f ? 6u : 4u;
    const float pathLength = max(layer.y - layer.x, 1.0f) / direction.y;
    const float stepLength = pathLength / float(stepCount);
    float opticalDepth = 0.0f;
    for (uint step = 0; step < stepCount; ++step) {
        opticalDepth +=
            sampleCloudDensityShadow(position, baseNoise, erosionNoise, curlNoise, weatherCloud,
                                     weatherLayer, weatherMotion, uniforms) *
            stepLength * 0.035f;
        position += direction * stepLength;
    }
    output.write(exp(-opticalDepth), gid);
}

vertex FullscreenVertex cloudCompositeVertex(uint vertexID [[vertex_id]]) {
    return fullscreenTriangle(vertexID);
}

fragment float4 cloudCompositeFragment(FullscreenVertex in [[stage_in]],
                                       texture2d<float> cloud [[texture(0)]],
                                       texture2d<float> cloudDepth [[texture(1)]],
                                       depth2d<float> sceneDepth [[texture(2)]],
                                       constant CloudRenderUniforms& uniforms [[buffer(0)]]) {
    const float opaqueDistance = sceneDistance(in.uv, sceneDepth, uniforms);
    const float2 size = float2(cloud.get_width(), cloud.get_height());
    const float2 texelPosition = in.uv * size - 0.5f;
    const int2 base = int2(floor(texelPosition));
    const float2 fraction = fract(texelPosition);
    const int2 extent = int2(cloud.get_width(), cloud.get_height());
    float4 result = 0.0f;
    float totalWeight = 0.0f;
    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            const uint2 pixel = uint2(clamp(base + int2(x, y), int2(0), extent - 1));
            const float hitDepth = cloudDepth.read(pixel).r;
            const float spatialWeight = (x == 0 ? 1.0f - fraction.x : fraction.x) *
                                        (y == 0 ? 1.0f - fraction.y : fraction.y);
            const float2 weights =
                cloudCompositeTapWeights(spatialWeight, hitDepth, opaqueDistance);
            result += cloud.read(pixel) * weights.x;
            totalWeight += weights.y;
        }
    }
    return totalWeight > 1.0e-5f ? result / totalWeight : 0.0f;
}

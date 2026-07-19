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

float3 viewPosition(float2 uv, float depth, constant IndirectLightingUniforms& uniforms) {
    float4 clip = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f);
    float4 view = uniforms.invProjection * clip;
    return view.xyz / max(view.w, 1.0e-6f);
}

float3 viewPositionFromLinearDepth(float2 uv, float linearDepth,
                                   constant IndirectLightingUniforms& uniforms) {
    float4 farClip = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 1.0f, 1.0f);
    float4 farView = uniforms.invProjection * farClip;
    float3 ray = farView.xyz / max(farView.w, 1.0e-6f);
    return ray * (linearDepth / max(abs(ray.z), 1.0e-5f));
}

float luminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 viewNormalFromDepth(float2 uv, texture2d<float> linearDepth, sampler depthSampler,
                           constant IndirectLightingUniforms& uniforms) {
    const float2 texel = 1.0f / float2(linearDepth.get_width(), linearDepth.get_height());
    const float centerDepth = linearDepth.sample(depthSampler, uv, level(0.0f)).r;
    const float3 center = viewPositionFromLinearDepth(uv, centerDepth, uniforms);
    const float2 leftUv = uv - float2(texel.x, 0.0f);
    const float2 rightUv = uv + float2(texel.x, 0.0f);
    const float2 upUv = uv - float2(0.0f, texel.y);
    const float2 downUv = uv + float2(0.0f, texel.y);
    const float leftDepth = linearDepth.sample(depthSampler, leftUv, level(0.0f)).r;
    const float rightDepth = linearDepth.sample(depthSampler, rightUv, level(0.0f)).r;
    const float upDepth = linearDepth.sample(depthSampler, upUv, level(0.0f)).r;
    const float downDepth = linearDepth.sample(depthSampler, downUv, level(0.0f)).r;
    const float3 horizontal =
        abs(rightDepth - centerDepth) <= abs(leftDepth - centerDepth)
            ? viewPositionFromLinearDepth(rightUv, rightDepth, uniforms) - center
            : center - viewPositionFromLinearDepth(leftUv, leftDepth, uniforms);
    const float3 vertical = abs(downDepth - centerDepth) <= abs(upDepth - centerDepth)
                                ? viewPositionFromLinearDepth(downUv, downDepth, uniforms) - center
                                : center - viewPositionFromLinearDepth(upUv, upDepth, uniforms);
    float3 normal = cross(horizontal, vertical);
    const float normalLengthSquared = dot(normal, normal);
    const float centerLengthSquared = dot(center, center);
    if (!(normalLengthSquared > 1.0e-8f)) {
        return centerLengthSquared > 1.0e-8f ? normalize(-center) : float3(0.0f, 0.0f, 1.0f);
    }
    normal *= rsqrt(normalLengthSquared);
    // Screen-space finite differences have an arbitrary winding at a grazing angle. The
    // visible surface must face the camera, not merely have a positive view-Z
    // component, otherwise floor normals flip while looking down a cave.
    return dot(normal, -center) < 0.0f ? -normal : normal;
}

float2 encodeOctahedralNormal(float3 normal) {
    normal /= max(abs(normal.x) + abs(normal.y) + abs(normal.z), 1.0e-6f);
    float2 encoded = normal.xy;
    if (normal.z < 0.0f) {
        const float2 signs = select(float2(-1.0f), float2(1.0f), encoded >= 0.0f);
        encoded = (1.0f - abs(encoded.yx)) * signs;
    }
    return encoded * 0.5f + 0.5f;
}

float3 decodeOctahedralNormal(float2 encoded) {
    const float2 projected = encoded * 2.0f - 1.0f;
    float3 normal = float3(projected, 1.0f - abs(projected.x) - abs(projected.y));
    const float fold = clamp(-normal.z, 0.0f, 1.0f);
    normal.xy += select(float2(-fold), float2(fold), normal.xy < 0.0f);
    const float lengthSquared = dot(normal, normal);
    return lengthSquared > 1.0e-8f ? normal * rsqrt(lengthSquared) : float3(0.0f, 0.0f, 1.0f);
}

float3 viewNormalFromGuide(float2 uv, texture2d<half, access::sample> normalGuide,
                           sampler normalSampler) {
    return decodeOctahedralNormal(float2(normalGuide.sample(normalSampler, uv).xy));
}

} // namespace

kernel void screenSpaceLinearDepthKernel(depth2d<float, access::read> sceneDepth [[texture(0)]],
                                         texture2d<float, access::write> linearDepth [[texture(1)]],
                                         constant IndirectLightingUniforms& uniforms [[buffer(0)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= linearDepth.get_width() || gid.y >= linearDepth.get_height()) {
        return;
    }
    const float depth = sceneDepth.read(gid);
    const float2 uv =
        (float2(gid) + 0.5f) / float2(linearDepth.get_width(), linearDepth.get_height());
    const float linear = depth >= 1.0f ? 65504.0f : abs(viewPosition(uv, depth, uniforms).z);
    linearDepth.write(linear, gid);
}

kernel void screenSpaceNormalKernel(texture2d<float, access::sample> linearDepth [[texture(0)]],
                                    texture2d<half, access::write> normalGuide [[texture(1)]],
                                    constant IndirectLightingUniforms& uniforms [[buffer(0)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= normalGuide.get_width() || gid.y >= normalGuide.get_height()) {
        return;
    }
    constexpr sampler pointSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    const float2 uv =
        (float2(gid) + 0.5f) / float2(normalGuide.get_width(), normalGuide.get_height());
    const float centerDepth = linearDepth.sample(pointSampler, uv, level(0.0f)).r;
    const float3 normal = centerDepth > INDIRECT_SKY_LINEAR_DEPTH
                              ? float3(0.0f, 0.0f, 1.0f)
                              : viewNormalFromDepth(uv, linearDepth, pointSampler, uniforms);
    normalGuide.write(half4(half2(encodeOctahedralNormal(normal)), half(0.0f), half(1.0f)), gid);
}

kernel void screenSpaceDepthReduceKernel(texture2d<float, access::read_write> linearDepth
                                         [[texture(0)]],
                                         constant uint& sourceLevel [[buffer(0)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    const uint destinationLevel = sourceLevel + 1u;
    const uint destinationWidth = max(linearDepth.get_width(destinationLevel), 1u);
    const uint destinationHeight = max(linearDepth.get_height(destinationLevel), 1u);
    if (gid.x >= destinationWidth || gid.y >= destinationHeight) {
        return;
    }
    const uint sourceWidth = max(linearDepth.get_width(sourceLevel), 1u);
    const uint sourceHeight = max(linearDepth.get_height(sourceLevel), 1u);
    const uint2 source = gid * 2u;
    // A halved odd dimension drops its trailing row or column. Fold it into
    // the boundary texel: the pyramid must stay a conservative minimum or the
    // Hi-Z march can prove a cell empty whose nearest surface lives in the
    // dropped edge.
    const uint sampleWidth = (gid.x == destinationWidth - 1u && (sourceWidth & 1u) != 0u) ? 3u : 2u;
    const uint sampleHeight =
        (gid.y == destinationHeight - 1u && (sourceHeight & 1u) != 0u) ? 3u : 2u;
    float minimumDepth = INFINITY;
    for (uint y = 0u; y < sampleHeight; ++y) {
        for (uint x = 0u; x < sampleWidth; ++x) {
            const uint2 coordinate =
                min(source + uint2(x, y), uint2(sourceWidth - 1u, sourceHeight - 1u));
            minimumDepth = min(minimumDepth, linearDepth.read(coordinate, sourceLevel).r);
        }
    }
    linearDepth.write(minimumDepth, gid, destinationLevel);
}

kernel void screenSpaceTraceKernel(texture2d<float, access::sample> linearDepth [[texture(0)]],
                                   texture2d<float, access::sample> radiance [[texture(1)]],
                                   texture2d<half, access::sample> normalGuide [[texture(2)]],
                                   texture2d<float, access::write> output [[texture(3)]],
                                   constant IndirectLightingUniforms& uniforms [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    constexpr sampler pointSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    const float2 workSize = float2(output.get_width(), output.get_height());
    const float2 uv = (float2(gid) + 0.5f) / workSize;
    const float centerDepth = linearDepth.sample(pointSampler, uv, level(0.0f)).r;
    if (centerDepth > INDIRECT_SKY_LINEAR_DEPTH) {
        output.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);
        return;
    }

    const float3 origin = viewPositionFromLinearDepth(uv, centerDepth, uniforms);
    const float3 normal = viewNormalFromGuide(uv, normalGuide, pointSampler);

    const uint quality = uint(uniforms.resolutionAndQuality.z + 0.5f);
    const uint rayCount = quality >= 2 ? INDIRECT_HIGH_RAY_COUNT : INDIRECT_MEDIUM_RAY_COUNT;
    const uint iterationCap =
        quality >= 2 ? INDIRECT_HIGH_HIZ_ITERATION_CAP : INDIRECT_MEDIUM_HIZ_ITERATION_CAP;
    // Radii and thickness are physical view-space distances in blocks. Rays
    // march the min-depth pyramid in texel space, so resolution and zoom
    // change only the traversal cost, never which geometry a ray can reach.
    const float aoRadius = max(uniforms.traceParams.x, 0.5f);
    const float thickness = max(uniforms.traceParams.y, 0.05f);
    const float giMaxDistance = max(uniforms.filterParams.x, aoRadius);
    const float2 pixelNoise =
        float2(fract(sin(dot(float2(gid), float2(12.9898f, 78.233f))) * 43758.5453f),
               fract(sin(dot(float2(gid), float2(39.3467f, 11.135f))) * 24634.6345f));
    const uint frameIndex = uint(max(uniforms.ambientAndFrame.w, 0.0f));
    const float2 fullSize = float2(linearDepth.get_width(), linearDepth.get_height());
    // The ray origin lifts off the receiver along its normal so the first
    // mip-zero lookup cannot re-test the receiver's own depth as a blocker.
    const float3 rayOrigin = origin + normal * max(0.02f, centerDepth * 0.005f);

    float occlusion = 0.0f;
    float3 bounce = 0.0f;
    for (uint rayIndex = 0; rayIndex < rayCount; ++rayIndex) {
        const float2 xi =
            screenSpaceRaySequenceSample(frameIndex * rayCount + rayIndex, pixelNoise);
        const float3 direction = screenSpaceCosineHemisphereDirection(xi, normal);

        // Clip the ray in view space so it never crosses the camera plane.
        float rayLength = giMaxDistance;
        if (direction.z > 1.0e-5f) {
            rayLength = min(rayLength, (-1.0e-3f - rayOrigin.z) / direction.z);
        }
        if (rayLength <= 1.0e-3f) {
            continue;
        }
        const float3 endPoint = screenSpaceTraceViewSample(rayOrigin, direction, rayLength);
        const float2 startPx =
            screenSpaceProjectViewPosition(rayOrigin, uniforms.projection) * fullSize;
        const float2 endPx =
            screenSpaceProjectViewPosition(endPoint, uniforms.projection) * fullSize;
        const float2 deltaPx = endPx - startPx;
        const float segmentLength = length(deltaPx);
        // A ray nearly perpendicular to the screen has no texels to march;
        // it can neither occlude nor find a visible source.
        if (segmentLength < 0.5f) {
            continue;
        }
        const float2 directionPx = deltaPx / segmentLength;
        const float startDepth = abs(rayOrigin.z);
        const float endDepth = abs(endPoint.z);

        // Bound travel to the screen so clamp_to_edge cannot fabricate hits.
        float maxTravel = segmentLength;
        if (directionPx.x > 1.0e-6f) {
            maxTravel = min(maxTravel, (fullSize.x - startPx.x) / directionPx.x);
        } else if (directionPx.x < -1.0e-6f) {
            maxTravel = min(maxTravel, -startPx.x / directionPx.x);
        }
        if (directionPx.y > 1.0e-6f) {
            maxTravel = min(maxTravel, (fullSize.y - startPx.y) / directionPx.y);
        } else if (directionPx.y < -1.0e-6f) {
            maxTravel = min(maxTravel, -startPx.y / directionPx.y);
        }

        // Skip the receiver's own mip-zero texel before classifying cells.
        float traveled = screenSpaceHiZCellExit(startPx, directionPx, 1.0f);
        uint mip = INDIRECT_HIZ_START_MIP;
        bool rayHit = false;
        float2 hitUv = uv;
        float hitSurfaceDepth = centerDepth;
        for (uint iteration = 0; iteration < iterationCap; ++iteration) {
            if (traveled >= maxTravel) {
                break;
            }
            const float2 samplePx = startPx + directionPx * traveled;
            const float2 sampleUv = samplePx / fullSize;
            const float cellSize = exp2(float(mip));
            const float cellExit =
                traveled + screenSpaceHiZCellExit(samplePx, directionPx, cellSize);
            const float entryDepth =
                screenSpaceHiZRayDepth(traveled / segmentLength, startDepth, endDepth);
            const float exitDepth = screenSpaceHiZRayDepth(min(cellExit, maxTravel) / segmentLength,
                                                           startDepth, endDepth);
            const float cellMinDepth =
                linearDepth.sample(pointSampler, sampleUv, level(float(mip))).r;
            if (screenSpaceHiZAdvances(entryDepth, exitDepth, cellMinDepth)) {
                traveled = cellExit;
                mip = min(mip + 1u, INDIRECT_HIZ_MAX_MIP);
                continue;
            }
            if (mip > 0u) {
                --mip;
                continue;
            }
            if (screenSpaceHiZSurfaceHit(exitDepth, cellMinDepth, thickness)) {
                rayHit = true;
                hitUv = sampleUv;
                hitSurfaceDepth = cellMinDepth;
                break;
            }
            // The ray passed behind a surface thinner than the thickness
            // interval or grazes it from the front. Step past this texel and
            // let the hierarchy re-prove empty space from one level up.
            traveled = cellExit;
            mip = 1u;
        }
        if (!rayHit) {
            continue;
        }

        const float3 hitPosition = viewPositionFromLinearDepth(hitUv, hitSurfaceDepth, uniforms);
        const float hitDistance = length(hitPosition - origin);
        occlusion += screenSpaceOcclusionFalloff(hitDistance, aoRadius);
        const float3 sourceNormal = viewNormalFromGuide(hitUv, normalGuide, pointSampler);
        const float sourceWeight = screenSpaceBounceSourceWeight(dot(sourceNormal, -direction),
                                                                 hitDistance, giMaxDistance);
        if (sourceWeight > 1.0e-5f) {
            // HDR radiance already contains the source surface's baked
            // ambient response. Applying accessibility here a second time
            // would erase emissive bounce and double-darken AO.
            bounce += radiance.sample(pointSampler, hitUv, level(0.0f)).rgb * sourceWeight;
        }
    }
    // Normalize against the fixed ray budget, not only successful hits. The
    // cosine-weighted density already carries the Lambert term, so the mean
    // of hit radiances is the unbiased one-bounce estimate.
    const float ao =
        saturate(1.0f - uniforms.traceParams.z * occlusion / max(float(rayCount), 1.0f));
    bounce *= uniforms.traceParams.w / max(float(rayCount), 1.0f);
    output.write(float4(max(bounce, 0.0f), ao), gid);
}

kernel void screenSpaceTemporalKernel(texture2d<float, access::read> current [[texture(0)]],
                                      texture2d<float, access::sample> linearDepth [[texture(1)]],
                                      texture2d<float, access::sample> previousHistory
                                      [[texture(2)]],
                                      texture2d<float, access::sample> previousDepth [[texture(3)]],
                                      texture2d<float, access::write> output [[texture(4)]],
                                      depth2d<float> sceneDepth [[texture(5)]],
                                      texture2d<half, access::sample> normalGuide [[texture(6)]],
                                      texture2d<float, access::sample> previousMoments
                                      [[texture(7)]],
                                      texture2d<float, access::write> outputMoments [[texture(8)]],
                                      constant IndirectLightingUniforms& uniforms [[buffer(0)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    constexpr sampler pointSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    const float2 size = float2(output.get_width(), output.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;

    // Firefly clamp before any statistics or accumulation: one lucky ray
    // that found a sunlit texel cannot seed the history with unbounded
    // energy. The clamp preserves hue by scaling all three channels.
    float4 center = current.read(gid);
    const float fireflyScale =
        screenSpaceFireflyClampScale(luminance(center.rgb), uniforms.temporalParams.w);
    center.rgb *= fireflyScale;

    // The 3x3 neighborhood feeds statistics only; spatial smoothing belongs
    // to the a-trous passes that follow. Depth and normal weights keep a
    // separate voxel face from widening this pixel's clamp window.
    const float currentDepth = linearDepth.sample(pointSampler, uv, level(0.0f)).r;
    const float3 centerNormal = viewNormalFromGuide(uv, normalGuide, pointSampler);
    float4 weightedSum = 0.0f;
    float4 weightedSquares = 0.0f;
    float weightSum = 0.0f;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const int2 coordinate = clamp(int2(gid) + int2(x, y), int2(0),
                                          int2(output.get_width() - 1, output.get_height() - 1));
            float4 value = current.read(uint2(coordinate));
            value.rgb *=
                screenSpaceFireflyClampScale(luminance(value.rgb), uniforms.temporalParams.w);
            const float2 neighborUv = (float2(coordinate) + 0.5f) / size;
            const float neighborDepth = linearDepth.sample(pointSampler, neighborUv, level(0.0f)).r;
            const float3 neighborNormal =
                viewNormalFromGuide(neighborUv, normalGuide, pointSampler);
            // Same edge stops as the a-trous passes with the luminance term
            // neutralized, so one tolerance retune reaches every consumer.
            const float weight =
                (x == 0 && y == 0 ? 2.0f : 1.0f) *
                screenSpaceAtrousEdgeWeight(neighborDepth - currentDepth,
                                            screenSpaceBilateralLinearDepthTolerance(currentDepth),
                                            dot(centerNormal, neighborNormal), 0.0f, 1.0f);
            weightedSum += value * weight;
            weightedSquares += value * value * weight;
            weightSum += weight;
        }
    }
    const float4 spatialMean = weightSum > 1.0e-5f ? weightedSum / weightSum : center;
    const float4 spatialSquares =
        weightSum > 1.0e-5f ? weightedSquares / weightSum : center * center;
    const float4 spatialSigma = sqrt(max(spatialSquares - spatialMean * spatialMean, float4(0.0f)));
    const float spatialLuminanceVariance =
        screenSpaceLuminanceVariance(luminance(spatialMean.rgb), luminance(spatialSquares.rgb));

    const float hardwareDepth = sceneDepth.sample(pointSampler, uv);
    const float4 worldH = uniforms.invViewProjection *
                          float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, hardwareDepth, 1.0f);
    const float4 previousClip =
        uniforms.previousViewProjection * float4(worldH.xyz / max(worldH.w, 1.0e-6f), 1.0f);
    const float previousW = max(previousClip.w, 1.0e-6f);
    const float previousDeviceDepth = previousClip.z / previousW;
    const float2 previousUv = previousClip.xy / previousW * float2(0.5f, -0.5f) + 0.5f;
    const bool inside = previousClip.w > 0.0f && all(previousUv >= 0.0f) &&
                        all(previousUv <= 1.0f) && previousDeviceDepth >= 0.0f &&
                        previousDeviceDepth <= 1.0f;
    bool valid = uniforms.resolutionAndQuality.w > 0.5f && inside;
    if (valid) {
        const float oldDepth = previousDepth.sample(pointSampler, previousUv).r;
        const float expectedDepth = abs(viewPosition(previousUv, previousDeviceDepth, uniforms).z);
        const float tolerance = screenSpaceTemporalLinearDepthTolerance(expectedDepth);
        valid = isfinite(oldDepth) && isfinite(expectedDepth) && tolerance > 0.0f &&
                abs(oldDepth - expectedDepth) <= tolerance;
    }

    // Age restarts at disocclusion, ramps the blend back up over about nine
    // frames, and saturates so stale history can never dominate forever.
    const float previousAge =
        valid ? max(previousMoments.sample(linearSampler, previousUv).z, 0.0f) : 0.0f;
    const float historyWeight =
        screenSpaceTemporalBlendWeight(previousAge, uniforms.temporalParams.x);
    const float age = min(previousAge + 1.0f, INDIRECT_HISTORY_MAX_AGE);

    // Accumulate raw luminance moments for the variance estimate that guides
    // both the history clamp and the a-trous filter width.
    const float centerLuminance = luminance(center.rgb);
    const float2 previousRawMoments =
        valid ? previousMoments.sample(linearSampler, previousUv).xy : float2(0.0f);
    const float firstMoment = mix(centerLuminance, previousRawMoments.x, historyWeight);
    const float secondMoment =
        mix(centerLuminance * centerLuminance, previousRawMoments.y, historyWeight);
    const float temporalVariance = screenSpaceLuminanceVariance(firstMoment, secondMoment);
    const float variance = screenSpaceVarianceForAge(temporalVariance, spatialLuminanceVariance,
                                                     age, uniforms.filterParams.z);
    outputMoments.write(float4(firstMoment, secondMoment, age, variance), gid);

    float4 result = center;
    if (valid) {
        // Variance-scaled clamp: a converged neighborhood collapses a stale
        // ghost to the floor in one frame, while a genuinely sparse bright
        // source keeps a wide clamp through its accumulated deviation. The
        // AO channel uses a tighter gamma so disocclusion darkening reacts
        // immediately.
        const float temporalSigma = sqrt(temporalVariance);
        const float4 reprojectedHistory = previousHistory.sample(linearSampler, previousUv);
        const float3 colorHalfRange =
            float3(screenSpaceVarianceClampHalfRange(max(spatialSigma.x, temporalSigma),
                                                     uniforms.temporalParams.y, 0.001f),
                   screenSpaceVarianceClampHalfRange(max(spatialSigma.y, temporalSigma),
                                                     uniforms.temporalParams.y, 0.001f),
                   screenSpaceVarianceClampHalfRange(max(spatialSigma.z, temporalSigma),
                                                     uniforms.temporalParams.y, 0.001f));
        const float aoHalfRange =
            screenSpaceVarianceClampHalfRange(spatialSigma.a, uniforms.temporalParams.z, 0.001f);
        const float4 clampedHistory = float4(
            clamp(reprojectedHistory.rgb, spatialMean.rgb - colorHalfRange,
                  spatialMean.rgb + colorHalfRange),
            clamp(reprojectedHistory.a, spatialMean.a - aoHalfRange, spatialMean.a + aoHalfRange));
        result = mix(center, clampedHistory, historyWeight);
    }
    output.write(max(result, float4(0.0f)), gid);
}

kernel void screenSpaceAtrousKernel(texture2d<float, access::read> source [[texture(0)]],
                                    texture2d<float, access::sample> linearDepth [[texture(1)]],
                                    texture2d<half, access::sample> normalGuide [[texture(2)]],
                                    texture2d<float, access::read> momentsAge [[texture(3)]],
                                    texture2d<float, access::write> destination [[texture(4)]],
                                    constant IndirectLightingUniforms& uniforms [[buffer(0)]],
                                    constant uint& stepSize [[buffer(1)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= destination.get_width() || gid.y >= destination.get_height()) {
        return;
    }
    constexpr sampler pointSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    const float2 size = float2(destination.get_width(), destination.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;
    const float4 center = source.read(gid);
    const float centerDepth = linearDepth.sample(pointSampler, uv, level(0.0f)).r;
    if (centerDepth > INDIRECT_SKY_LINEAR_DEPTH) {
        destination.write(center, gid);
        return;
    }
    const float3 centerNormal = viewNormalFromGuide(uv, normalGuide, pointSampler);
    const float centerLuminance = luminance(center.rgb);
    // The variance written by the temporal pass is read once per frame; each
    // wider wavelet iteration divides the luminance sigma by its step so the
    // filter tightens as its footprint grows instead of washing out edges.
    const float variance = max(momentsAge.read(gid).w, 0.0f);
    const float luminanceSigma =
        max(uniforms.filterParams.y * sqrt(variance) / float(max(stepSize, 1u)), 1.0e-4f);
    const float depthTolerance =
        screenSpaceBilateralLinearDepthTolerance(centerDepth) * float(max(stepSize, 1u));

    float3 colorSum = 0.0f;
    float colorWeightSum = 0.0f;
    float aoSum = 0.0f;
    float aoWeightSum = 0.0f;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const int2 coordinate =
                clamp(int2(gid) + int2(x, y) * int(stepSize), int2(0),
                      int2(destination.get_width() - 1, destination.get_height() - 1));
            const float4 value = source.read(uint2(coordinate));
            const float2 neighborUv = (float2(coordinate) + 0.5f) / size;
            const float neighborDepth = linearDepth.sample(pointSampler, neighborUv, level(0.0f)).r;
            const float3 neighborNormal =
                viewNormalFromGuide(neighborUv, normalGuide, pointSampler);
            const float kernelWeight = (x == 0 ? 2.0f : 1.0f) * (y == 0 ? 2.0f : 1.0f) / 16.0f;
            const float normalDot = dot(centerNormal, neighborNormal);
            const float colorWeight =
                kernelWeight *
                screenSpaceAtrousEdgeWeight(neighborDepth - centerDepth, depthTolerance, normalDot,
                                            luminance(value.rgb) - centerLuminance, luminanceSigma);
            colorSum += value.rgb * colorWeight;
            colorWeightSum += colorWeight;
            // Ambient occlusion is low-frequency by construction and must
            // not chase luminance edges in the colored bounce, so its weight
            // drops the luminance term.
            const float aoWeight =
                kernelWeight * screenSpaceAtrousEdgeWeight(neighborDepth - centerDepth,
                                                           depthTolerance, normalDot, 0.0f, 1.0f);
            aoSum += value.a * aoWeight;
            aoWeightSum += aoWeight;
        }
    }
    const float3 filteredColor = colorWeightSum > 1.0e-5f ? colorSum / colorWeightSum : center.rgb;
    const float filteredAo = aoWeightSum > 1.0e-5f ? aoSum / aoWeightSum : center.a;
    destination.write(float4(max(filteredColor, 0.0f), saturate(filteredAo)), gid);
}

kernel void screenSpaceHistoryDepthKernel(texture2d<float, access::sample> linearDepth
                                          [[texture(0)]],
                                          texture2d<float, access::write> historyDepth
                                          [[texture(1)]],
                                          constant IndirectLightingUniforms& uniforms [[buffer(0)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= historyDepth.get_width() || gid.y >= historyDepth.get_height()) {
        return;
    }
    constexpr sampler pointSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    const float2 uv =
        (float2(gid) + 0.5f) / float2(historyDepth.get_width(), historyDepth.get_height());
    // Keep history in linear view depth. Device depth loses most of its
    // precision near the far side of a grazing floor and makes a stable
    // camera look like a per-pixel disocclusion.
    historyDepth.write(linearDepth.sample(pointSampler, uv, level(0.0f)).r, gid);
}

vertex FullscreenVertex screenSpaceApplyVertex(uint vertexID [[vertex_id]]) {
    return fullscreenTriangle(vertexID);
}

fragment float4 screenSpaceApplyFragment(FullscreenVertex in [[stage_in]],
                                         texture2d<float> indirect [[texture(0)]],
                                         texture2d<float> surface [[texture(1)]],
                                         depth2d<float> sceneDepth [[texture(2)]],
                                         texture2d<half> normalGuide [[texture(3)]],
                                         constant IndirectLightingUniforms& uniforms
                                         [[buffer(0)]]) {
    constexpr sampler pointSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
    // Never interpolate native surface data across a block edge. The indirect
    // history is intentionally lower resolution, so it receives a manual
    // bilinear reconstruction whose candidates must agree with this receiver.
    const float receiverDeviceDepth = sceneDepth.sample(pointSampler, in.uv);
    if (receiverDeviceDepth >= 0.99999f) {
        return 0.0f;
    }
    const float4 surfaceSample = surface.sample(pointSampler, in.uv);
    float4 indirectSample = float4(0.0f, 0.0f, 0.0f, 1.0f);

    // Quality off binds the 1x1 neutral texture. Leave the neutral value
    // explicit so the ambient-only path remains valid without history data.
    if (uniforms.resolutionAndQuality.z > 0.5f) {
        const int2 indirectExtent = int2(int(indirect.get_width()), int(indirect.get_height()));
        const float2 indirectSize = float2(indirectExtent);
        const float2 texelPosition = in.uv * indirectSize - 0.5f;
        const int2 baseCoordinate = int2(floor(texelPosition));
        const float2 fraction = fract(texelPosition);
        const float receiverDepth = abs(viewPosition(in.uv, receiverDeviceDepth, uniforms).z);
        const float3 receiverNormal = viewNormalFromGuide(in.uv, normalGuide, pointSampler);
        float4 accumulated = 0.0f;
        float weightSum = 0.0f;
        for (int y = 0; y < 2; ++y) {
            const float yWeight = y == 0 ? 1.0f - fraction.y : fraction.y;
            for (int x = 0; x < 2; ++x) {
                const float xWeight = x == 0 ? 1.0f - fraction.x : fraction.x;
                const int2 coordinate =
                    clamp(baseCoordinate + int2(x, y), int2(0), indirectExtent - int2(1));
                const float2 candidateUv = (float2(coordinate) + 0.5f) / indirectSize;
                const float candidateDeviceDepth = sceneDepth.sample(pointSampler, candidateUv);
                if (candidateDeviceDepth >= 0.99999f) {
                    continue;
                }
                const float candidateDepth =
                    abs(viewPosition(candidateUv, candidateDeviceDepth, uniforms).z);
                const float weight =
                    xWeight * yWeight *
                    screenSpaceJointBilateralUpsampleWeight(
                        receiverDepth, candidateDepth, receiverNormal,
                        viewNormalFromGuide(candidateUv, normalGuide, pointSampler));
                accumulated += indirect.sample(pointSampler, candidateUv) * weight;
                weightSum += weight;
            }
        }
        if (weightSum > 1.0e-5f) {
            indirectSample = accumulated / weightSum;
        } else {
            // A half-resolution trace center can land entirely on the other
            // side of a one-voxel edge. Do not borrow that perpendicular
            // face: find the closest compatible trace owner in the immediate
            // 3x3 neighborhood and use it without blending. This fallback is
            // intentionally limited to the no-owner case, so the ordinary
            // joint bilateral path remains smooth and perpendicular or
            // depth-discontinuous faces cannot leak a bright cave-floor
            // bounce across their edge.
            const int2 nearestCoordinate =
                clamp(int2(floor(texelPosition + 0.5f)), int2(0), indirectExtent - int2(1));
            float bestScore = 0.0f;
            float4 fallback = indirectSample;
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    const int2 coordinate =
                        clamp(nearestCoordinate + int2(x, y), int2(0), indirectExtent - int2(1));
                    const float2 candidateUv = (float2(coordinate) + 0.5f) / indirectSize;
                    const float candidateDeviceDepth = sceneDepth.sample(pointSampler, candidateUv);
                    if (candidateDeviceDepth >= 0.99999f) {
                        continue;
                    }
                    const float candidateDepth =
                        abs(viewPosition(candidateUv, candidateDeviceDepth, uniforms).z);
                    const float compatibility = screenSpaceJointBilateralUpsampleWeight(
                        receiverDepth, candidateDepth, receiverNormal,
                        viewNormalFromGuide(candidateUv, normalGuide, pointSampler));
                    const float2 offset = float2(coordinate) - texelPosition;
                    const float score = compatibility / (1.0f + dot(offset, offset));
                    if (score > bestScore) {
                        bestScore = score;
                        fallback = indirect.sample(pointSampler, candidateUv);
                    }
                }
            }
            if (bestScore > 1.0e-5f) {
                indirectSample = fallback;
            }
        }
    }
    const float3 ambient = uniforms.ambientAndFrame.xyz;
    const float3 ambientCorrection =
        ambient * surfaceSample.rgb * surfaceSample.a * (indirectSample.a - 1.0f);
    const float3 bounce = indirectSample.rgb * surfaceSample.rgb;
    return float4(ambientCorrection + bounce, 0.0f);
}

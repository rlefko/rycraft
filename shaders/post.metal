#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Final composite, the one linear-HDR → display conversion.
//
// Samples the HDR scene and the (half-res) bloom pyramid, then applies:
//   exposure → bloom add → Hable filmic tonemap → vibrance grade →
//   optional CAS sharpen → dither → BGRA8 drawable.
//
// This pass ALWAYS runs, with bloom disabled the renderer binds a 4×4
// black fallback so the frame is still tonemapped (the old pipeline blitted
// raw scene colors whenever bloom was off).
// ---------------------------------------------------------------------------

struct PostVertexOut {
    float4 clipPosition [[position]];
    float2 vUV;
};

vertex PostVertexOut postCompositeVertex(uint vertexID [[vertex_id]]) {
    // Fullscreen triangle; V flipped so sampling preserves orientation
    // (texture v runs down, NDC y runs up).
    const float2 positions[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    PostVertexOut out;
    out.clipPosition = float4(positions[vertexID], 0.0f, 1.0f);
    out.vUV = float2(positions[vertexID].x * 0.5f + 0.5f, 0.5f - positions[vertexID].y * 0.5f);
    return out;
}

// Hable "Uncharted 2" filmic tonemap: a strong shoulder that keeps rolling
// far past display max (a 12x HDR sun still lands on a gradient, where the
// previous Uchimura shoulder plateaued to flat white within a few stops) and
// a filmic toe that holds blacks down so dark scenes need less exposure lift.
// Standard constants; W is the linear white point the curve normalizes to.
static float hableCurve(float x) {
    constexpr float A = 0.15f; // shoulder strength
    constexpr float B = 0.50f; // linear strength
    constexpr float C = 0.10f; // linear angle
    constexpr float D = 0.20f; // toe strength
    constexpr float E = 0.02f; // toe numerator
    constexpr float F = 0.30f; // toe denominator
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

static float3 filmic3(float3 x) {
    constexpr float W = 11.2f; // linear white point
    const float whiteScale = 1.0f / hableCurve(W);
    return float3(hableCurve(x.r), hableCurve(x.g), hableCurve(x.b)) * whiteScale;
}

// Rec.709 luminance, the shared luma everywhere in the post stack.
static float luma(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

// Vibrance: boost saturation where there is little, spare already-saturated
// pixels (a plain saturation knob pushes skies and foliage to neon).
static float3 applyVibrance(float3 c, float vibrance) {
    float mx = max3(c.r, c.g, c.b);
    float mn = min3(c.r, c.g, c.b);
    float sat = mx - mn;
    float boost = vibrance * (1.0f - saturate(sat));
    return mix(float3(luma(c)), c, 1.0f + boost);
}

// Exposure + bloom + tonemap + grade for one sample position, CAS needs
// the display-referred value of each tap, so the whole chain is reused.
static float3 displayColor(texture2d<float> scene, texture2d<float> bloom, sampler s, float2 uv,
                           constant PostUniforms& post, float exposure) {
    float3 hdr = scene.sample(s, uv).rgb;
    hdr += bloom.sample(s, uv).rgb * post.bloomIntensity;
    hdr *= exposure * post.exposure;
    // The filmic curve maps ~2x brighter input to the same display value as
    // the old curve's mids, so a fixed gain keeps daylight where it was.
    float3 mapped = filmic3(max(hdr, 0.0f) * 2.0f);
    return applyVibrance(mapped, post.vibrance - 1.0f);
}

// Frame-offset wrapper over the shared IGN (shader_types.hpp), deterministic
// per pixel/frame, breaks 8-bit banding in sky gradients.
static float interleavedGradientNoise(float2 px, uint frame) {
    return interleavedGradientNoise(px + float2(frame % 64u) * 5.588238f);
}

// ---------------------------------------------------------------------------
// Lens flare, occlusion probe + procedural ghosts.
//
// The probe kernel (one thread, 16 depth taps in a small grid around the
// sun's screen position) eases FlareState.visibility toward the fraction of
// taps that see sky, so the flare dims smoothly as terrain slides across the
// sun. The composite then draws four chromatic ghosts marching along the
// sun→center line plus a soft halo, all scaled by visibility × strength.
// ---------------------------------------------------------------------------
kernel void flareProbe(depth2d<float> sceneDepth [[texture(0)]],
                       texture2d<float> cloudTransmittance [[texture(1)]],
                       device FlareState& flare [[buffer(0)]],
                       constant PostUniforms& post [[buffer(1)]]) {
    constexpr sampler depthPoint(mag_filter::nearest, min_filter::nearest, address::clamp_to_edge);
    float visible = 0.0f;
    // 4x4 tap grid, ~1.5% of the screen across
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            float2 offset = (float2(float(x), float(y)) - 1.5f) * 0.005f;
            float2 uv = post.sunScreenUV + offset;
            if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
                continue; // off-screen taps count as occluded
            }
            if (sceneDepth.sample(depthPoint, uv) >= 1.0f) {
                visible += 1.0f / 16.0f;
            }
        }
    }
    constexpr sampler cloudSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float4 cloud = cloudTransmittance.sample(cloudSampler, post.sunScreenUV);
    const float cloudVisibility =
        post.flareCloudOpacityTexture != 0u ? 1.0f - saturate(cloud.a) : saturate(cloud.r);
    visible *= cloudVisibility;
    // Ease so the flare fades over ~10 frames instead of popping
    flare.visibility = mix(flare.visibility, visible, 0.15f);
}

// Four ghosts + a halo along the sun→center axis (classic flare chain).
static float3 lensFlareGhosts(float2 uv, constant PostUniforms& post) {
    const float aspect = post.resolution.x / max(post.resolution.y, 1.0f);
    const float2 sun = post.sunScreenUV;
    const float2 axis = float2(0.5f) - sun; // toward the screen center

    const float offsets[4] = {0.6f, 1.1f, 1.7f, 2.3f};
    const float sizes[4] = {0.050f, 0.032f, 0.075f, 0.045f};
    const float3 tints[4] = {float3(1.0f, 0.75f, 0.45f), float3(0.5f, 0.8f, 1.0f),
                             float3(1.0f, 0.55f, 0.35f), float3(0.7f, 1.0f, 0.75f)};

    // Per-ghost and halo brightness are kept deliberately low (0.14 / 0.05) so
    // the flare hints at the sun rather than washing the frame; the composite
    // scales them again by sun intensity × occlusion.
    float3 result = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float2 pos = sun + axis * offsets[i];
        float2 d = (uv - pos) * float2(aspect, 1.0f);
        float g = pow(saturate(1.0f - length(d) / sizes[i]), 2.5f);
        result += tints[i] * (g * 0.14f);
    }
    // Soft halo hugging the sun itself
    float2 dSun = (uv - sun) * float2(aspect, 1.0f);
    result +=
        float3(1.0f, 0.85f, 0.6f) * (pow(saturate(1.0f - length(dSun) / 0.22f), 3.0f) * 0.05f);
    return result;
}

fragment float4 postCompositeFragment(PostVertexOut in [[stage_in]],
                                      texture2d<float> sceneTexture [[texture(0)]],
                                      texture2d<float> bloomTexture [[texture(1)]],
                                      constant PostUniforms& post [[buffer(0)]],
                                      constant ExposureState& exposureState [[buffer(1)]],
                                      constant FlareState& flare [[buffer(2)]]) {
    constexpr sampler linearSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float exposure = exposureState.exposure;

    float3 color = displayColor(sceneTexture, bloomTexture, linearSampler, in.vUV, post, exposure);

    // CAS-style adaptive sharpen: 4 cross taps, each run through the same
    // display transform (sharpening must operate on what the eye sees).
    if (post.sharpening > 0.0f) {
        float2 texel = 1.0f / post.resolution;
        float3 n = displayColor(sceneTexture, bloomTexture, linearSampler,
                                in.vUV + float2(0.0f, -texel.y), post, exposure);
        float3 s = displayColor(sceneTexture, bloomTexture, linearSampler,
                                in.vUV + float2(0.0f, texel.y), post, exposure);
        float3 w = displayColor(sceneTexture, bloomTexture, linearSampler,
                                in.vUV + float2(-texel.x, 0.0f), post, exposure);
        float3 e = displayColor(sceneTexture, bloomTexture, linearSampler,
                                in.vUV + float2(texel.x, 0.0f), post, exposure);
        float3 minC = min(color, min(min(n, s), min(w, e)));
        float3 maxC = max(color, max(max(n, s), max(w, e)));
        // Weight from local contrast: flat areas sharpen, edges saturate less
        float3 amp = sqrt(saturate(min(minC, 1.0f - maxC) / max(maxC, 1e-4f)));
        float3 weight = amp * -0.2f * post.sharpening;
        color = saturate((color + (n + s + w + e) * weight) / (1.0f + 4.0f * weight));
    }

    // Lens flare ghosts, gated by the occlusion probe (post-tonemap so the
    // overlay's brightness doesn't ride the auto-exposure).
    float flareGate = post.flareStrength * flare.visibility;
    if (flareGate > 0.001f) {
        color += lensFlareGhosts(in.vUV, post) * flareGate;
    }

    // Sub-LSB dither before the 8-bit quantization
    color += (interleavedGradientNoise(in.clipPosition.xy, post.frameIndex) - 0.5f) / 255.0f;

    return float4(saturate(color), 1.0f);
}

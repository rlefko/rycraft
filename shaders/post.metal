#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Final composite — the one linear-HDR → display conversion.
//
// Samples the HDR scene and the (half-res) bloom pyramid, then applies:
//   exposure → bloom add → Uchimura tonemap → vibrance grade →
//   optional CAS sharpen → dither → BGRA8 drawable.
//
// This pass ALWAYS runs — with bloom disabled the renderer binds a 4×4
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

// Uchimura 2017 "Gran Turismo" tonemap: a linear section through the mids
// (the vibrant look keeps its saturation there), smooth toe and shoulder.
// P = 1 (display max), a = 1 (linear slope), m = 0.22 (linear start),
// l = 0.30 (linear length), c = 1.33 (black tightness), b = 0.
static float uchimura(float x) {
    // l = 0.30 starts the shoulder earlier than the stock 0.4, compressing
    // more highlight range below display max so the sun disc keeps its limb
    // gradient instead of plateauing flat white.
    constexpr float P = 1.0f, a = 1.0f, m = 0.22f, l = 0.30f, c = 1.33f, b = 0.0f;
    const float l0 = ((P - m) * l) / a;
    const float S0 = m + l0;
    const float S1 = m + a * l0;
    const float C2 = (a * P) / (P - S1);
    const float CP = -C2 / P;

    float w0 = 1.0f - smoothstep(0.0f, m, x);
    float w2 = step(m + l0, x);
    float w1 = 1.0f - w0 - w2;

    float T = m * pow(x / m, c) + b;             // toe
    float L = m + a * (x - m);                   // linear
    float S = P - (P - S1) * exp(CP * (x - S0)); // shoulder

    return T * w0 + L * w1 + S * w2;
}

static float3 uchimura3(float3 x) {
    return float3(uchimura(x.r), uchimura(x.g), uchimura(x.b));
}

// Rec.709 luminance — the shared luma everywhere in the post stack.
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

// Exposure + bloom + tonemap + grade for one sample position — CAS needs
// the display-referred value of each tap, so the whole chain is reused.
static float3 displayColor(texture2d<float> scene, texture2d<float> bloom, sampler s, float2 uv,
                           constant PostUniforms& post, float exposure) {
    float3 hdr = scene.sample(s, uv).rgb;
    hdr += bloom.sample(s, uv).rgb * post.bloomIntensity;
    hdr *= exposure * post.exposure;
    float3 mapped = uchimura3(max(hdr, 0.0f));
    return applyVibrance(mapped, post.vibrance - 1.0f);
}

// Frame-offset wrapper over the shared IGN (shader_types.hpp) — deterministic
// per pixel/frame, breaks 8-bit banding in sky gradients.
static float interleavedGradientNoise(float2 px, uint frame) {
    return interleavedGradientNoise(px + float2(frame % 64u) * 5.588238f);
}

// ---------------------------------------------------------------------------
// Lens flare — occlusion probe + procedural ghosts.
//
// The probe kernel (one thread, 16 depth taps in a small grid around the
// sun's screen position) eases FlareState.visibility toward the fraction of
// taps that see sky, so the flare dims smoothly as terrain slides across the
// sun. The composite then draws four chromatic ghosts marching along the
// sun→center line plus a soft halo, all scaled by visibility × strength.
// ---------------------------------------------------------------------------
kernel void flareProbe(depth2d<float> sceneDepth [[texture(0)]],
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

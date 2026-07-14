#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Auto-exposure — "Simple HDR" eye adaptation.
//
// One threadgroup samples the HDR scene on a sparse grid, reduces the mean
// log2 luminance in threadgroup memory, then thread 0 blends it into the
// persistent smoothedLogLum (temporal EMA — the eye adapts over time, not
// per frame) and derives an exposure that maps the scene's average toward
// middle grey. GPU-side; no CPU readback stalls the frame.
// ---------------------------------------------------------------------------

constant uint THREADS = 256; // 16×16 threadgroup

static float luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

kernel void exposureReduce(texture2d<float> scene [[texture(0)]],
                           device ExposureState& state [[buffer(0)]],
                           constant ExposureParams& params [[buffer(1)]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[THREADS];

    // Each thread samples one grid cell (params.sampleGrid.x × .y cells),
    // striding across the whole frame in normalized coords.
    uint gx = tid % params.sampleGrid.x;
    uint gy = tid / params.sampleGrid.x;
    float2 uv = (float2(gx, gy) + 0.5f) / float2(params.sampleGrid);

    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float lum = max(luminance(scene.sample(s, uv).rgb), 1e-5f);
    partial[tid] = clamp(log2(lum), params.minLogLum, params.maxLogLum);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel sum reduction over the threadgroup.
    for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float frameLogLum = partial[0] / float(THREADS);
        // Temporal adaptation: ease the stored average toward this frame's.
        float adapted = mix(state.smoothedLogLum, frameLogLum, params.adaptationRate);
        state.smoothedLogLum = adapted;
        // exposure = key / averageLuminance. The minExposure floor keeps
        // bright daylight from crushing (its average would otherwise pull
        // exposure far below 1); maxExposure lifts caves and night.
        float avgLum = exp2(adapted);
        state.exposure = clamp(params.keyValue / max(avgLum, 1e-4f), params.minExposure,
                               params.maxExposure);
    }
}

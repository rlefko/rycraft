#pragma once

#import <Metal/Metal.h>

#include "common/ema.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// GpuFrameTimer, real GPU time per frame, and per pass on request.
//
// Whole-frame timing reads the command buffer's GPUStartTime/GPUEndTime in a
// completed handler and is always on (the handler is the frame's only extra
// work). Per-pass timing samples MTLCounterSampleBuffer timestamps at stage
// boundaries and exists only when requested at construction AND the device
// supports stage-boundary sampling, otherwise attachPass() is a no-op and
// no sample buffers are allocated, so the disabled path costs nothing.
//
// Completed handlers outlive teardown races by capturing a shared_ptr to the
// mutable state instead of `this`.
// ---------------------------------------------------------------------------
class GpuFrameTimer {
public:
    GpuFrameTimer(id<MTLDevice> device, bool enablePassCounters);
    ~GpuFrameTimer();

    // Rotate to the next sample-buffer slot. Call once at the top of a frame.
    void beginFrame();

    // Reserve a start/end timestamp pair in this frame's sample buffer and
    // attach it to the pass. No-op when pass counters are off or the frame's
    // pass capacity is exhausted.
    void attachPass(MTLRenderPassDescriptor* desc, const char* label);

    // Timestamp a sequence of compute dispatches encoded between these two
    // calls. The returned token is opaque and may represent a disabled timer.
    uint32_t beginComputePass(id<MTLComputeCommandEncoder> encoder, const char* label);
    void endComputePass(id<MTLComputeCommandEncoder> encoder, uint32_t token);

    // Install the completed handler that records the frame's GPU ms and
    // resolves this frame's pass samples. Call once per frame before commit.
    void endFrame(id<MTLCommandBuffer> commandBuffer);

    // EMA of whole-frame GPU time, safe to read from the render thread.
    float frameMsEma() const { return state_->frameMs.value(); }

    // "scene 3.21 water 0.85 ui 0.05" from the last resolved frame; empty
    // when pass counters are off. Render-thread read, values lag one frame.
    std::string passBreakdown() const;

private:
    static constexpr uint32_t MAX_PASSES = 32;
    static constexpr uint32_t SLOTS = 3; // matches the deepest frames-in-flight

    struct PassSample {
        const char* label;   // static strings only
        uint32_t startIndex; // end timestamp lives at startIndex + 1
    };

    struct ResolvedPass {
        const char* label;
        float ms;
    };

    // State the completed handlers write; shared so a handler that fires
    // during teardown holds it alive.
    struct State {
        AtomicEmaMs frameMs;
        std::mutex resolvedMutex;
        std::vector<ResolvedPass> resolved;
    };

    id<MTLDevice> device_;
    std::shared_ptr<State> state_;
    id<MTLCounterSampleBuffer> sampleBuffers_[SLOTS] = {};
    std::vector<PassSample> framePasses_;
    uint32_t slot_ = 0;
    bool passCountersEnabled_ = false;
    bool renderCountersEnabled_ = false;
    bool computeCountersEnabled_ = false;

    // Calibration anchor for converting GPU timestamp ticks to nanoseconds.
    MTLTimestamp cpuAnchor_ = 0;
    MTLTimestamp gpuAnchor_ = 0;
};

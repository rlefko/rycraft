#import "render/gpu_timer.hpp"

#include "common/error.hpp"

#include <cstdio>

GpuFrameTimer::GpuFrameTimer(id<MTLDevice> device, bool enablePassCounters)
    : device_(device), state_(std::make_shared<State>()) {
    framePasses_.reserve(MAX_PASSES);
    if (!enablePassCounters) {
        return;
    }

    if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        RY_LOG_INFO("GPU pass counters requested but stage-boundary sampling is unsupported");
        return;
    }
    id<MTLCounterSet> timestampSet = nil;
    for (id<MTLCounterSet> set in device.counterSets) {
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            timestampSet = set;
            break;
        }
    }
    if (!timestampSet) {
        RY_LOG_INFO("GPU pass counters requested but no timestamp counter set exists");
        return;
    }

    MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor alloc] init];
    desc.counterSet = timestampSet;
    desc.sampleCount = MAX_PASSES * 2;
    desc.storageMode = MTLStorageModeShared;
    for (uint32_t i = 0; i < SLOTS; ++i) {
        NSError* error = nil;
        sampleBuffers_[i] = [device newCounterSampleBufferWithDescriptor:desc error:&error];
        if (!sampleBuffers_[i]) {
            RY_LOG_INFO("GPU pass counters unavailable (sample buffer creation failed)");
            return;
        }
    }
    [device sampleTimestamps:&cpuAnchor_ gpuTimestamp:&gpuAnchor_];
    passCountersEnabled_ = true;
}

void GpuFrameTimer::beginFrame() {
    if (!passCountersEnabled_) {
        return;
    }
    slot_ = (slot_ + 1) % SLOTS;
    framePasses_.clear();
}

void GpuFrameTimer::attachPass(MTLRenderPassDescriptor* desc, const char* label) {
    if (!passCountersEnabled_ || framePasses_.size() >= MAX_PASSES) {
        return;
    }
    const uint32_t startIndex = static_cast<uint32_t>(framePasses_.size()) * 2;
    desc.sampleBufferAttachments[0].sampleBuffer = sampleBuffers_[slot_];
    desc.sampleBufferAttachments[0].startOfVertexSampleIndex = startIndex;
    desc.sampleBufferAttachments[0].endOfVertexSampleIndex = MTLCounterDontSample;
    desc.sampleBufferAttachments[0].startOfFragmentSampleIndex = MTLCounterDontSample;
    desc.sampleBufferAttachments[0].endOfFragmentSampleIndex = startIndex + 1;
    framePasses_.push_back({label, startIndex});
}

void GpuFrameTimer::endFrame(id<MTLCommandBuffer> commandBuffer) {
    std::shared_ptr<State> state = state_;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        state->frameMs.record(static_cast<float>((cb.GPUEndTime - cb.GPUStartTime) * 1000.0));
    }];

    if (!passCountersEnabled_ || framePasses_.empty()) {
        return;
    }

    // Capture this frame's reservations and the slot's buffer by value; the
    // handler resolves timestamps only after the GPU finished the frame.
    id<MTLCounterSampleBuffer> sampleBuffer = sampleBuffers_[slot_];
    std::vector<PassSample> passes = framePasses_;
    id<MTLDevice> device = device_;
    const MTLTimestamp cpuAnchor = cpuAnchor_;
    const MTLTimestamp gpuAnchor = gpuAnchor_;

    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        NSData* data = [sampleBuffer resolveCounterRange:NSMakeRange(0, passes.size() * 2)];
        if (!data) {
            return;
        }
        const auto* stamps = static_cast<const MTLCounterResultTimestamp*>(data.bytes);

        // GPU timestamp units are unspecified; scale ticks to nanoseconds
        // from two paired CPU/GPU samples (CPU side is nanoseconds).
        MTLTimestamp cpuNow = 0;
        MTLTimestamp gpuNow = 0;
        [device sampleTimestamps:&cpuNow gpuTimestamp:&gpuNow];
        if (gpuNow <= gpuAnchor) {
            return;
        }
        const double nsPerTick =
            static_cast<double>(cpuNow - cpuAnchor) / static_cast<double>(gpuNow - gpuAnchor);

        std::lock_guard<std::mutex> lock(state->resolvedMutex);
        state->resolved.clear();
        for (const PassSample& pass : passes) {
            const MTLTimestamp start = stamps[pass.startIndex].timestamp;
            const MTLTimestamp end = stamps[pass.startIndex + 1].timestamp;
            const float ms =
                (end > start) ? static_cast<float>((end - start) * nsPerTick / 1.0e6) : 0.f;
            state->resolved.push_back({pass.label, ms});
        }
    }];
}

std::string GpuFrameTimer::passBreakdown() const {
    if (!passCountersEnabled_) {
        return {};
    }
    std::lock_guard<std::mutex> lock(state_->resolvedMutex);
    std::string out;
    char buf[48];
    for (const ResolvedPass& pass : state_->resolved) {
        snprintf(buf, sizeof(buf), "%s%s %.2f", out.empty() ? "" : " ", pass.label, pass.ms);
        out += buf;
    }
    return out;
}

#pragma once

#import <Metal/Metal.h>

#include "common/error.hpp"

#include <atomic>
#include <cstring>
#include <memory>

// ---------------------------------------------------------------------------
// FrameRing — frames-in-flight gate + per-frame constants arena.
//
// The CPU may encode up to SLOTS frames before the GPU finishes the oldest
// one, so every buffer the CPU rewrites per frame must come from a slot the
// GPU is no longer reading. One shared arena buffer is split into SLOTS
// regions; alloc() linearly sub-allocates 256-byte-aligned ranges from the
// current frame's region, and the semaphore guarantees the GPU released it.
// (The UI overlay predates this and keeps its own 3-slot vertex ring.)
//
// completedFrame() feeds the MegaBuffer's deferred-free drain: a chunk mesh
// region freed during frame N may still be read by frames < N, so it is
// recycled only once completedFrame() >= N.
// ---------------------------------------------------------------------------
class FrameRing {
public:
    static constexpr uint32_t SLOTS = 3;

    // One CPU-visible allocation within the current frame's slot.
    struct Alloc {
        void* ptr = nullptr;
        id<MTLBuffer> buffer = nil;
        uint64_t offset = 0;
    };

    FrameRing(id<MTLDevice> device, uint64_t slotBytes)
        : state_(std::make_shared<State>())
        , slotBytes_(slotBytes) {
        arena_ =
            [device newBufferWithLength:slotBytes * SLOTS options:MTLResourceStorageModeShared];
        if (!arena_) {
            RY_LOG_FATAL("Failed to allocate the frame-constants arena");
        }
        state_->semaphore = dispatch_semaphore_create(SLOTS);
    }

    // Block until the GPU has released a slot, then claim it for this frame.
    void waitAndBegin() {
        dispatch_semaphore_wait(state_->semaphore, DISPATCH_TIME_FOREVER);
        ++frameIndex_;
        cursor_ = 0;
    }

    // 256-byte-aligned sub-allocation from the current slot. Exhausting the
    // slot is a sizing bug, not a runtime condition — fail loudly.
    Alloc alloc(uint64_t size) {
        const uint64_t offset = (cursor_ + 255) & ~uint64_t{255};
        if (offset + size > slotBytes_) {
            RY_LOG_FATAL("Frame-constants slot exhausted — grow FrameRing slotBytes");
        }
        cursor_ = offset + size;
        const uint64_t slotBase = (frameIndex_ % SLOTS) * slotBytes_;
        return {static_cast<uint8_t*>(arena_.contents) + slotBase + offset, arena_,
                slotBase + offset};
    }

    Alloc push(const void* data, uint64_t size) {
        Alloc a = alloc(size);
        std::memcpy(a.ptr, data, size);
        return a;
    }

    // Signal the slot free and record frame completion when the GPU is done.
    // Pair with exactly one waitAndBegin() per frame, before commit. The
    // handler captures the shared state, not `this` — a frame completing
    // after teardown must not write into a destroyed FrameRing (the same
    // rule GpuFrameTimer::State exists for).
    void signalOnCompletion(id<MTLCommandBuffer> commandBuffer) {
        std::shared_ptr<State> state = state_;
        const uint64_t frame = frameIndex_;
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
          state->completedFrame.store(frame, std::memory_order_release);
          dispatch_semaphore_signal(state->semaphore);
        }];
    }

    // Release the claimed slot without encoding (a frame abandoned after
    // waitAndBegin — e.g. an encoder failed). Nothing references the slot.
    void cancelFrame() { dispatch_semaphore_signal(state_->semaphore); }

    uint64_t frameIndex() const { return frameIndex_; }
    uint64_t completedFrame() const {
        return state_->completedFrame.load(std::memory_order_acquire);
    }

private:
    // Written by GPU completion handlers, which may outlive the FrameRing.
    struct State {
        dispatch_semaphore_t semaphore = nil;
        std::atomic<uint64_t> completedFrame{0};
    };

    id<MTLBuffer> arena_ = nil;
    std::shared_ptr<State> state_;
    uint64_t slotBytes_ = 0;
    uint64_t cursor_ = 0;
    uint64_t frameIndex_ = 0; // first frame is 1; 0 means "before any frame"
};

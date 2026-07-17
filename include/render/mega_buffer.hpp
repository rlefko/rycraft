#pragma once

#include "render/vertex.hpp"
#import <Metal/Metal.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

class MegaBuffer {
public:
    struct ChunkAllocation {
        id<MTLBuffer> vertexBuffer;
        id<MTLBuffer> indexBuffer;
        uint64_t vertexOffset;
        uint64_t indexOffset;
        uint32_t vertexCount;
        uint32_t indexCount;
    };

    static constexpr uint64_t ALIGNMENT = 256;

    explicit MegaBuffer(id<MTLDevice> device, uint64_t vertexSize = 128 * 1024 * 1024,
                        uint64_t indexSize = 64 * 1024 * 1024);

    ChunkAllocation allocate(uint32_t vertexCount, uint32_t indexCount);

    void uploadVertices(const void* data, size_t size, uint64_t offset);
    void uploadIndices(const void* data, size_t size, uint64_t offset);

    void free(ChunkAllocation& alloc);

    // Frames-in-flight variant of free(): the region stays allocated until
    // drainDeferredFrees learns the GPU finished every frame that could
    // still read it. A region freed while encoding frame N was last drawn
    // in an earlier frame, so it recycles once completedFrame >= N.
    void deferFree(ChunkAllocation& alloc, uint64_t frame);
    void drainDeferredFrees(uint64_t completedFrame);

    uint64_t vertexUsed() const;
    uint64_t indexUsed() const;
    bool owns(const ChunkAllocation& alloc) const {
        return alloc.vertexBuffer == _vertexBuffer && alloc.indexBuffer == _indexBuffer;
    }
    uint64_t vertexCapacity() const { return _vertexSize; }
    uint64_t indexCapacity() const { return _indexSize; }

    // Sort {offset, size} regions and merge adjacent ones in place. Public
    // static so tests can pin it without a Metal device — the previous
    // in-line version wrote one element past the end of the vector on a
    // single-entry list (slow heap corruption: garbled audio, then malloc
    // traps minutes later) and dropped the last region on every pass.
    static void coalesceFreeList(std::vector<std::pair<uint64_t, uint64_t>>& freeList);

private:
    static uint64_t alignUp(uint64_t value);

    id<MTLBuffer> _vertexBuffer;
    id<MTLBuffer> _indexBuffer;
    uint64_t _vertexSize;
    uint64_t _indexSize;
    uint64_t _vertexPtr;
    uint64_t _indexPtr;
    std::vector<std::pair<uint64_t, uint64_t>> _vertexFreeList;
    std::vector<std::pair<uint64_t, uint64_t>> _indexFreeList;
    struct DeferredFree {
        ChunkAllocation alloc;
        uint64_t frame; // render frame during which the region was freed
    };
    std::vector<DeferredFree> _deferredFrees;
    mutable std::mutex _mutex;

    // free() body without the lock. Deferred drains append a whole completed
    // frame's regions before coalescing once, avoiding one free-list sort per
    // retired cube during movement.
    void freeLocked(ChunkAllocation& alloc, bool coalesce);

    bool tryBumpAllocate(uint64_t& outOffset, uint64_t alignedSize, uint64_t bufferSize,
                         uint64_t& bumpPtr) const;

    bool tryFreeListAllocate(uint64_t& outOffset, uint64_t alignedSize, uint64_t bufferSize,
                             std::vector<std::pair<uint64_t, uint64_t>>& freeList);
};

// A bounded collection of lazily allocated Metal buffer pairs. Large horizon
// residency cannot rely on one multi-gigabyte MTLBuffer being available as a
// single contiguous allocation. Each returned ChunkAllocation already carries
// its concrete buffers, so draw submission stays unchanged while frees and
// uploads route back to the owning slab.
class SegmentedMegaBuffer {
public:
    using ChunkAllocation = MegaBuffer::ChunkAllocation;

    SegmentedMegaBuffer(id<MTLDevice> device, uint64_t vertexCapacity, uint64_t indexCapacity,
                        uint64_t vertexSlabSize, uint64_t indexSlabSize);

    ChunkAllocation allocate(uint32_t vertexCount, uint32_t indexCount);
    void uploadVertices(const void* data, size_t size, const ChunkAllocation& alloc);
    void uploadIndices(const void* data, size_t size, const ChunkAllocation& alloc);
    void free(ChunkAllocation& alloc);
    void deferFree(ChunkAllocation& alloc, uint64_t frame);
    void drainDeferredFrees(uint64_t completedFrame);

    uint64_t vertexUsed() const;
    uint64_t indexUsed() const;
    uint64_t vertexCapacity() const { return _vertexCapacity; }
    uint64_t indexCapacity() const { return _indexCapacity; }
    size_t segmentCount() const { return _segments.size(); }

private:
    MegaBuffer& owner(const ChunkAllocation& alloc) const;
    bool canCreateSegment() const;
    MegaBuffer& createSegment();

    id<MTLDevice> _device;
    uint64_t _vertexCapacity;
    uint64_t _indexCapacity;
    uint64_t _vertexSlabSize;
    uint64_t _indexSlabSize;
    uint64_t _createdVertexCapacity = 0;
    uint64_t _createdIndexCapacity = 0;
    std::vector<std::unique_ptr<MegaBuffer>> _segments;
};

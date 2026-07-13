#pragma once

#include "render/vertex.hpp"
#import <Metal/Metal.h>
#include <cstdint>
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

    uint64_t vertexUsed() const;
    uint64_t indexUsed() const;
    uint64_t vertexCapacity() const { return _vertexSize; }
    uint64_t indexCapacity() const { return _indexSize; }

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
    mutable std::mutex _mutex;

    bool tryBumpAllocate(uint64_t& outOffset, uint64_t alignedSize, uint64_t bufferSize,
                         uint64_t& bumpPtr) const;

    bool tryFreeListAllocate(uint64_t& outOffset, uint64_t alignedSize, uint64_t bufferSize,
                             std::vector<std::pair<uint64_t, uint64_t>>& freeList);
};

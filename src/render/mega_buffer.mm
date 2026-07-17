#include "render/mega_buffer.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>

uint64_t MegaBuffer::alignUp(uint64_t value) {
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

bool MegaBuffer::tryBumpAllocate(uint64_t& outOffset, uint64_t alignedSize, uint64_t bufferSize,
                                 uint64_t& bumpPtr) const {
    uint64_t end = bumpPtr + alignedSize;
    if (end > bufferSize) {
        return false;
    }
    outOffset = bumpPtr;
    bumpPtr = end;
    return true;
}

bool MegaBuffer::tryFreeListAllocate(uint64_t& outOffset, uint64_t alignedSize,
                                     uint64_t /*bufferSize*/,
                                     std::vector<std::pair<uint64_t, uint64_t>>& freeList) {
    // The list stays sorted and coalesced when frees are published, so this is
    // deterministic first-fit without an O(n log n) sort on every allocation.
    for (auto it = freeList.begin(); it != freeList.end(); ++it) {
        uint64_t regionStart = it->first;
        uint64_t regionSize = it->second;

        if (regionSize < alignedSize) {
            continue;
        }

        // Ensure the allocation start is aligned within this region
        uint64_t allocStart = (regionStart + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        uint64_t available = regionStart + regionSize - allocStart;

        if (available < alignedSize) {
            continue;
        }

        outOffset = allocStart;

        // Remove used region from free list
        it = freeList.erase(it);

        // Add leftover region back if large enough
        uint64_t leftoverStart = allocStart + alignedSize;
        uint64_t leftoverSize = regionStart + regionSize - leftoverStart;
        if (leftoverSize >= ALIGNMENT) {
            freeList.insert(it, {leftoverStart, leftoverSize});
        }

        return true;
    }

    return false;
}

MegaBuffer::MegaBuffer(id<MTLDevice> device, uint64_t vertexSize, uint64_t indexSize)
    : _vertexSize(vertexSize), _indexSize(indexSize), _vertexPtr(0), _indexPtr(0) {
    _vertexBuffer = [device newBufferWithLength:static_cast<NSUInteger>(vertexSize)
                                        options:MTLResourceStorageModeShared];
    if (!_vertexBuffer) {
        throw std::runtime_error("Metal vertex buffer allocation failed");
    }

    _indexBuffer = [device newBufferWithLength:static_cast<NSUInteger>(indexSize)
                                       options:MTLResourceStorageModeShared];
    if (!_indexBuffer) {
        throw std::runtime_error("Metal index buffer allocation failed");
    }
}

MegaBuffer::ChunkAllocation MegaBuffer::allocate(uint32_t vertexCount, uint32_t indexCount) {
    if (vertexCount == 0 && indexCount == 0) {
        throw std::invalid_argument("cannot allocate zero-sized chunk");
    }

    std::lock_guard lock(_mutex);

    uint64_t vertexBytes = vertexCount ? alignUp(vertexCount * sizeof(Vertex)) : 0;
    uint64_t indexBytes = indexCount ? alignUp(indexCount * sizeof(uint32_t)) : 0;

    uint64_t vertexOffset = 0;
    uint64_t indexOffset = 0;

    // Vertex allocation: try bump pointer first, then free list
    bool vertexAllocated = false;
    bool vertexFromBump = false;
    if (vertexBytes > 0) {
        vertexAllocated = tryBumpAllocate(vertexOffset, vertexBytes, _vertexSize,
                                          const_cast<uint64_t&>(_vertexPtr));
        vertexFromBump = vertexAllocated;
        if (!vertexAllocated) {
            vertexAllocated =
                tryFreeListAllocate(vertexOffset, vertexBytes, _vertexSize, _vertexFreeList);
        }
    }

    // Index allocation: try bump pointer first, then free list
    bool indexAllocated = false;
    bool indexFromBump = false;
    if (indexBytes > 0) {
        indexAllocated =
            tryBumpAllocate(indexOffset, indexBytes, _indexSize, const_cast<uint64_t&>(_indexPtr));
        indexFromBump = indexAllocated;
        if (!indexAllocated) {
            indexAllocated =
                tryFreeListAllocate(indexOffset, indexBytes, _indexSize, _indexFreeList);
        }
    }

    if ((vertexBytes > 0 && !vertexAllocated) || (indexBytes > 0 && !indexAllocated)) {
        const auto rollback = [](uint64_t offset, uint64_t size, bool fromBump, uint64_t& bumpPtr,
                                 std::vector<std::pair<uint64_t, uint64_t>>& freeList) {
            if (size == 0)
                return;
            if (fromBump && offset + size == bumpPtr) {
                bumpPtr = offset;
                return;
            }
            freeList.push_back({offset, size});
            coalesceFreeList(freeList);
        };
        if (vertexAllocated) {
            rollback(vertexOffset, vertexBytes, vertexFromBump, _vertexPtr, _vertexFreeList);
        }
        if (indexAllocated) {
            rollback(indexOffset, indexBytes, indexFromBump, _indexPtr, _indexFreeList);
        }
        throw std::runtime_error(
            "mega buffer allocation failed: insufficient space in vertex or index buffer");
    }

    return ChunkAllocation{
        .vertexBuffer = _vertexBuffer,
        .indexBuffer = _indexBuffer,
        .vertexOffset = vertexOffset,
        .indexOffset = indexOffset,
        .vertexCount = vertexCount,
        .indexCount = indexCount,
    };
}

void MegaBuffer::uploadVertices(const void* data, size_t size, uint64_t offset) {
    if (!data) {
        throw std::invalid_argument("null vertex data pointer");
    }
    if (offset + size > _vertexSize) {
        throw std::out_of_range("vertex upload exceeds buffer bounds");
    }
    std::memcpy(static_cast<uint8_t*>([_vertexBuffer contents]) + offset, data, size);
}

void MegaBuffer::uploadIndices(const void* data, size_t size, uint64_t offset) {
    if (!data) {
        throw std::invalid_argument("null index data pointer");
    }
    if (offset + size > _indexSize) {
        throw std::out_of_range("index upload exceeds buffer bounds");
    }
    std::memcpy(static_cast<uint8_t*>([_indexBuffer contents]) + offset, data, size);
}

void MegaBuffer::coalesceFreeList(std::vector<std::pair<uint64_t, uint64_t>>& freeList) {
    if (freeList.empty()) {
        return;
    }
    std::sort(freeList.begin(), freeList.end());
    // Canonical compaction: the first element is both the first read and
    // the first write slot; writeIt always points at the last kept region.
    auto writeIt = freeList.begin();
    for (auto readIt = std::next(freeList.begin()); readIt != freeList.end(); ++readIt) {
        if (writeIt->first + writeIt->second == readIt->first) {
            writeIt->second += readIt->second;
        } else {
            *(++writeIt) = *readIt;
        }
    }
    freeList.erase(std::next(writeIt), freeList.end());
}

void MegaBuffer::free(ChunkAllocation& alloc) {
    std::lock_guard lock(_mutex);
    freeLocked(alloc, true);
}

void MegaBuffer::deferFree(ChunkAllocation& alloc, uint64_t frame) {
    std::lock_guard lock(_mutex);
    _deferredFrees.push_back({alloc, frame});
    alloc.vertexBuffer = nil;
    alloc.indexBuffer = nil;
    alloc.vertexOffset = 0;
    alloc.indexOffset = 0;
    alloc.vertexCount = 0;
    alloc.indexCount = 0;
}

void MegaBuffer::drainDeferredFrees(uint64_t completedFrame) {
    std::lock_guard lock(_mutex);
    bool released = false;
    std::erase_if(_deferredFrees, [&](DeferredFree& deferred) {
        if (deferred.frame > completedFrame) {
            return false;
        }
        freeLocked(deferred.alloc, false);
        released = true;
        return true;
    });
    if (released) {
        coalesceFreeList(_vertexFreeList);
        coalesceFreeList(_indexFreeList);
    }
}

void MegaBuffer::freeLocked(ChunkAllocation& alloc, bool coalesce) {
    if (alloc.vertexCount > 0) {
        uint64_t vertexBytes = alignUp(alloc.vertexCount * sizeof(Vertex));
        _vertexFreeList.push_back({alloc.vertexOffset, vertexBytes});
    }

    if (alloc.indexCount > 0) {
        uint64_t indexBytes = alignUp(alloc.indexCount * sizeof(uint32_t));
        _indexFreeList.push_back({alloc.indexOffset, indexBytes});
    }

    if (coalesce) {
        coalesceFreeList(_vertexFreeList);
        coalesceFreeList(_indexFreeList);
    }

    alloc.vertexBuffer = nil;
    alloc.indexBuffer = nil;
    alloc.vertexOffset = 0;
    alloc.indexOffset = 0;
    alloc.vertexCount = 0;
    alloc.indexCount = 0;
}

uint64_t MegaBuffer::vertexUsed() const {
    std::lock_guard lock(_mutex);
    uint64_t freeBytes = 0;
    for (const auto& [offset, size] : _vertexFreeList) {
        (void)offset;
        freeBytes += size;
    }
    return _vertexPtr - freeBytes;
}

uint64_t MegaBuffer::indexUsed() const {
    std::lock_guard lock(_mutex);
    uint64_t freeBytes = 0;
    for (const auto& [offset, size] : _indexFreeList) {
        (void)offset;
        freeBytes += size;
    }
    return _indexPtr - freeBytes;
}

SegmentedMegaBuffer::SegmentedMegaBuffer(id<MTLDevice> device, uint64_t vertexCapacity,
                                         uint64_t indexCapacity, uint64_t vertexSlabSize,
                                         uint64_t indexSlabSize)
    : _device(device), _vertexCapacity(vertexCapacity), _indexCapacity(indexCapacity),
      _vertexSlabSize(vertexSlabSize), _indexSlabSize(indexSlabSize) {
    if (!_device || vertexCapacity == 0 || indexCapacity == 0 || vertexSlabSize == 0 ||
        indexSlabSize == 0) {
        throw std::invalid_argument("segmented mega buffer capacities must be positive");
    }
    const uint64_t vertexSegments = (vertexCapacity + vertexSlabSize - 1) / vertexSlabSize;
    const uint64_t indexSegments = (indexCapacity + indexSlabSize - 1) / indexSlabSize;
    if (vertexSegments != indexSegments) {
        throw std::invalid_argument("segmented mega buffer slab counts must match");
    }
    _segments.reserve(static_cast<size_t>(vertexSegments));
}

bool SegmentedMegaBuffer::canCreateSegment() const {
    return _createdVertexCapacity < _vertexCapacity && _createdIndexCapacity < _indexCapacity;
}

MegaBuffer& SegmentedMegaBuffer::createSegment() {
    if (!canCreateSegment()) {
        throw std::runtime_error("segmented mega buffer capacity exhausted");
    }
    const uint64_t vertexSize = std::min(_vertexSlabSize, _vertexCapacity - _createdVertexCapacity);
    const uint64_t indexSize = std::min(_indexSlabSize, _indexCapacity - _createdIndexCapacity);
    auto segment = std::make_unique<MegaBuffer>(_device, vertexSize, indexSize);
    _createdVertexCapacity += vertexSize;
    _createdIndexCapacity += indexSize;
    _segments.push_back(std::move(segment));
    return *_segments.back();
}

MegaBuffer::ChunkAllocation SegmentedMegaBuffer::allocate(uint32_t vertexCount,
                                                          uint32_t indexCount) {
    const uint64_t vertexBytes = static_cast<uint64_t>(vertexCount) * sizeof(Vertex);
    const uint64_t indexBytes = static_cast<uint64_t>(indexCount) * sizeof(uint32_t);
    if (vertexBytes > _vertexSlabSize || indexBytes > _indexSlabSize) {
        throw std::runtime_error("far mesh exceeds one segmented arena slab");
    }
    for (const auto& segment : _segments) {
        try {
            return segment->allocate(vertexCount, indexCount);
        } catch (const std::runtime_error&) {
            // A full or fragmented slab is expected during horizon churn.
        }
    }
    return createSegment().allocate(vertexCount, indexCount);
}

MegaBuffer& SegmentedMegaBuffer::owner(const ChunkAllocation& alloc) const {
    for (const auto& segment : _segments) {
        if (segment->owns(alloc)) {
            return *segment;
        }
    }
    throw std::invalid_argument("allocation does not belong to segmented mega buffer");
}

void SegmentedMegaBuffer::uploadVertices(const void* data, size_t size,
                                         const ChunkAllocation& alloc) {
    owner(alloc).uploadVertices(data, size, alloc.vertexOffset);
}

void SegmentedMegaBuffer::uploadIndices(const void* data, size_t size,
                                        const ChunkAllocation& alloc) {
    owner(alloc).uploadIndices(data, size, alloc.indexOffset);
}

void SegmentedMegaBuffer::free(ChunkAllocation& alloc) {
    owner(alloc).free(alloc);
}

void SegmentedMegaBuffer::deferFree(ChunkAllocation& alloc, uint64_t frame) {
    owner(alloc).deferFree(alloc, frame);
}

void SegmentedMegaBuffer::drainDeferredFrees(uint64_t completedFrame) {
    for (const auto& segment : _segments) {
        segment->drainDeferredFrees(completedFrame);
    }
}

uint64_t SegmentedMegaBuffer::vertexUsed() const {
    uint64_t total = 0;
    for (const auto& segment : _segments)
        total += segment->vertexUsed();
    return total;
}

uint64_t SegmentedMegaBuffer::indexUsed() const {
    uint64_t total = 0;
    for (const auto& segment : _segments)
        total += segment->indexUsed();
    return total;
}

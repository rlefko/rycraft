#pragma once

#import <Metal/Metal.h>

#include "entity/item_entity.hpp"
#include "render/shader_types.hpp"

#include <functional>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// ItemEntityRenderer - draws dropped items as small spinning colored cubes.
//
// Sibling of EntityRenderer: it reuses the exact entity shader, vertex
// format, MSAA material pipeline, and cascade shadow contract. One 0.25-edge
// cube mesh per ItemType is baked lazily and cached (only the types actually
// dropped ever allocate).
// ---------------------------------------------------------------------------
class ItemEntityRenderer {
public:
    ItemEntityRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);
    ~ItemEntityRenderer();

    void render(id<MTLRenderCommandEncoder> encoder, id<MTLBuffer> uniformsBuffer,
                uint64_t uniformsOffset, const std::vector<ItemEntity>& items,
                const std::function<bool(const AABB&)>& isVisible);
    void renderShadowCasters(id<MTLRenderCommandEncoder> encoder,
                             const std::vector<ItemEntity>& items,
                             const std::function<bool(const AABB&)>& isVisible);

private:
    struct Mesh {
        id<MTLBuffer> vertexBuffer;
        id<MTLBuffer> indexBuffer;
        uint32_t indexCount = 0;
    };

    const Mesh& meshFor(ItemType type);

    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;
    std::unordered_map<uint16_t, Mesh> _meshes; // keyed by ItemType id
};

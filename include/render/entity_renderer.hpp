#pragma once

#import <Metal/Metal.h>

#include "entity/entity.hpp"
#include "render/shader_types.hpp"

#include <functional>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// EntityRenderer — draws animals as their voxel-box models.
//
// Each EntityType's model (adult and baby variants) bakes into one small
// static mesh at construction; per entity, only a model matrix changes.
// Draws happen inside the main MSAA scene pass with the shared Uniforms
// buffer, so entities receive the same sun/ambient/fog treatment as terrain.
// ---------------------------------------------------------------------------
class EntityRenderer {
public:
    EntityRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);

    // Draw every visible entity. `isVisible` is the caller's frustum test.
    // The uniforms live in the caller's frame ring, hence buffer + offset.
    void render(id<MTLRenderCommandEncoder> encoder, id<MTLBuffer> uniformsBuffer,
                uint64_t uniformsOffset, const std::vector<std::shared_ptr<Entity>>& entities,
                const std::function<bool(const AABB&)>& isVisible);

private:
    struct Mesh {
        id<MTLBuffer> vertexBuffer;
        id<MTLBuffer> indexBuffer;
        uint32_t indexCount = 0;
    };

    static constexpr int TYPE_COUNT = 4; // SHEEP, COW, PIG, CHICKEN
    Mesh _meshes[TYPE_COUNT][2];         // [type][adult=0 / baby=1]

    id<MTLRenderPipelineState> _pipelineState;

    Mesh buildMesh(id<MTLDevice> device, EntityType type, bool isBaby);
};

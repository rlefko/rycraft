#include "render/entity_renderer.hpp"

#include "common/error.hpp"

#include <array>

// Append one axis-aligned colored box to the mesh. Boxes are centered in
// X/Z on `offset` and bottom-anchored in Y (matching Entity::getVoxelModel).
static void appendBox(std::vector<EntityVertex>& vertices, std::vector<uint16_t>& indices,
                      const VoxelBlock& block) {
    const float x0 = block.offset.x - block.size.x * 0.5f;
    const float x1 = block.offset.x + block.size.x * 0.5f;
    const float y0 = block.offset.y;
    const float y1 = block.offset.y + block.size.y;
    const float z0 = block.offset.z - block.size.z * 0.5f;
    const float z1 = block.offset.z + block.size.z * 0.5f;
    const simd_float3 color = simd_make_float3(block.color.x, block.color.y, block.color.z);

    struct Face {
        simd_float3 normal;
        simd_float3 corners[4];
    };
    const Face faces[6] = {
        {{1, 0, 0}, {{x1, y0, z0}, {x1, y1, z0}, {x1, y1, z1}, {x1, y0, z1}}},
        {{-1, 0, 0}, {{x0, y0, z1}, {x0, y1, z1}, {x0, y1, z0}, {x0, y0, z0}}},
        {{0, 0, 1}, {{x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}, {x0, y0, z1}}},
        {{0, 0, -1}, {{x0, y0, z0}, {x0, y1, z0}, {x1, y1, z0}, {x1, y0, z0}}},
        {{0, 1, 0}, {{x0, y1, z0}, {x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}}},
        {{0, -1, 0}, {{x0, y0, z1}, {x0, y0, z0}, {x1, y0, z0}, {x1, y0, z1}}},
    };

    for (const Face& face : faces) {
        auto base = static_cast<uint16_t>(vertices.size());
        for (const simd_float3& corner : face.corners) {
            vertices.push_back(EntityVertex{corner, face.normal, color});
        }
        indices.push_back(base);
        indices.push_back(static_cast<uint16_t>(base + 1));
        indices.push_back(static_cast<uint16_t>(base + 2));
        indices.push_back(base);
        indices.push_back(static_cast<uint16_t>(base + 2));
        indices.push_back(static_cast<uint16_t>(base + 3));
    }
}

EntityRenderer::Mesh EntityRenderer::buildMesh(id<MTLDevice> device, EntityType type,
                                               bool isBaby) {
    std::vector<EntityVertex> vertices;
    std::vector<uint16_t> indices;
    for (const VoxelBlock& block : Entity::getVoxelModel(type, isBaby)) {
        appendBox(vertices, indices, block);
    }

    Mesh mesh;
    mesh.vertexBuffer = [device newBufferWithBytes:vertices.data()
                                            length:vertices.size() * sizeof(EntityVertex)
                                           options:MTLResourceStorageModeShared];
    mesh.indexBuffer = [device newBufferWithBytes:indices.data()
                                           length:indices.size() * sizeof(uint16_t)
                                          options:MTLResourceStorageModeShared];
    mesh.indexCount = static_cast<uint32_t>(indices.size());
    if (!mesh.vertexBuffer || !mesh.indexBuffer) {
        RY_LOG_FATAL("Failed to allocate entity mesh buffers");
    }
    return mesh;
}

EntityRenderer::EntityRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary) {
    id<MTLFunction> vertexFunc = [shaderLibrary newFunctionWithName:@"entityVertexMain"];
    id<MTLFunction> fragmentFunc = [shaderLibrary newFunctionWithName:@"entityFragmentMain"];
    if (!vertexFunc || !fragmentFunc) {
        RY_LOG_FATAL("Failed to load entity shader functions");
    }

    auto pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    // Entities draw inside the 4x MSAA scene pass
    pipelineDesc.rasterSampleCount = 4;

    NSError* error = nil;
    _pipelineState = [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!_pipelineState) {
        RY_LOG_FATAL("Failed to create entity pipeline state");
    }

    for (int type = 0; type < TYPE_COUNT; ++type) {
        _meshes[type][0] = buildMesh(device, static_cast<EntityType>(type), false);
        _meshes[type][1] = buildMesh(device, static_cast<EntityType>(type), true);
    }
}

void EntityRenderer::render(id<MTLRenderCommandEncoder> encoder,
                            id<MTLBuffer> uniformsBuffer,
                            const std::vector<std::shared_ptr<Entity>>& entities,
                            const std::function<bool(const AABB&)>& isVisible) {
    if (entities.empty()) return;

    [encoder setRenderPipelineState:_pipelineState];
    [encoder setVertexBuffer:uniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:uniformsBuffer offset:0 atIndex:1];

    for (const auto& entity : entities) {
        if (!entity || !entity->alive) continue;
        if (!isVisible(entity->aabb)) continue;

        const int type = static_cast<int>(entity->type);
        if (type < 0 || type >= TYPE_COUNT) continue;
        const Mesh& mesh = _meshes[type][entity->isBaby ? 1 : 0];

        EntityModel model;
        model.model = matrix_identity_float4x4;
        model.model.columns[3] = simd_make_float4(entity->position.x, entity->position.y,
                                                  entity->position.z, 1.0f);
        [encoder setVertexBytes:&model length:sizeof(model) atIndex:2];

        [encoder setVertexBuffer:mesh.vertexBuffer offset:0 atIndex:0];
        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:mesh.indexCount
                             indexType:MTLIndexTypeUInt16
                           indexBuffer:mesh.indexBuffer
                     indexBufferOffset:0];
    }
}

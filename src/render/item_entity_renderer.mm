#include "render/item_entity_renderer.hpp"

#include "common/error.hpp"
#include "render/pixel_formats.hpp"

#include <cmath>
#include <vector>

const ItemEntityRenderer::Mesh& ItemEntityRenderer::meshFor(ItemType type) {
    const auto key = static_cast<uint16_t>(type);
    auto existing = _meshes.find(key);
    if (existing != _meshes.end()) {
        return existing->second;
    }

    const uint32_t rgb = itemSwatchColor(type);
    const simd_float3 color = simd_make_float3(static_cast<float>((rgb >> 16) & 0xFF) / 255.f,
                                               static_cast<float>((rgb >> 8) & 0xFF) / 255.f,
                                               static_cast<float>(rgb & 0xFF) / 255.f);
    const float half = ItemEntity::SIZE * 0.5f;
    const float y0 = -half;
    const float y1 = half;

    struct Face {
        simd_float3 normal;
        simd_float3 corners[4];
    };
    const Face faces[6] = {
        {{1, 0, 0}, {{half, y0, -half}, {half, y1, -half}, {half, y1, half}, {half, y0, half}}},
        {{-1, 0, 0},
         {{-half, y0, half}, {-half, y1, half}, {-half, y1, -half}, {-half, y0, -half}}},
        {{0, 0, 1}, {{half, y0, half}, {half, y1, half}, {-half, y1, half}, {-half, y0, half}}},
        {{0, 0, -1},
         {{-half, y0, -half}, {-half, y1, -half}, {half, y1, -half}, {half, y0, -half}}},
        {{0, 1, 0}, {{-half, y1, -half}, {-half, y1, half}, {half, y1, half}, {half, y1, -half}}},
        {{0, -1, 0}, {{-half, y0, half}, {-half, y0, -half}, {half, y0, -half}, {half, y0, half}}},
    };

    std::vector<EntityVertex> vertices;
    std::vector<uint16_t> indices;
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

    Mesh mesh;
    mesh.vertexBuffer = [_device newBufferWithBytes:vertices.data()
                                             length:vertices.size() * sizeof(EntityVertex)
                                            options:MTLResourceStorageModeShared];
    mesh.indexBuffer = [_device newBufferWithBytes:indices.data()
                                            length:indices.size() * sizeof(uint16_t)
                                           options:MTLResourceStorageModeShared];
    mesh.indexCount = static_cast<uint32_t>(indices.size());
    if (!mesh.vertexBuffer || !mesh.indexBuffer) {
        RY_LOG_FATAL("Failed to allocate item entity mesh buffers");
    }
    return _meshes.emplace(key, mesh).first->second;
}

ItemEntityRenderer::ItemEntityRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary)
    : _device(device) {
    id<MTLFunction> vertexFunc = [shaderLibrary newFunctionWithName:@"entityVertexMain"];
    id<MTLFunction> fragmentFunc = [shaderLibrary newFunctionWithName:@"entityFragmentMain"];
    if (!vertexFunc || !fragmentFunc) {
        RY_LOG_FATAL("Failed to load entity shader functions for item rendering");
    }

    auto pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    pipelineDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
    pipelineDesc.rasterSampleCount = 4;

    NSError* error = nil;
    _pipelineState = [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!_pipelineState) {
        RY_LOG_FATAL("Failed to create item entity pipeline state");
    }
}

void ItemEntityRenderer::render(id<MTLRenderCommandEncoder> encoder, id<MTLBuffer> uniformsBuffer,
                                uint64_t uniformsOffset, const std::vector<ItemEntity>& items,
                                const std::function<bool(const AABB&)>& isVisible) {
    if (items.empty())
        return;

    [encoder setRenderPipelineState:_pipelineState];
    [encoder setVertexBuffer:uniformsBuffer offset:uniformsOffset atIndex:1];
    [encoder setFragmentBuffer:uniformsBuffer offset:uniformsOffset atIndex:1];

    for (size_t index = 0; index < items.size(); ++index) {
        const ItemEntity& item = items[index];
        if (item.stack.empty())
            continue;
        if (!isVisible(item.getAABB()))
            continue;

        const Mesh& mesh = meshFor(item.stack.type);

        // Spin about Y and bob gently; the phase is deterministic from age
        // plus a per-item offset so a pile does not spin in lockstep.
        const float phase = static_cast<float>(item.ageTicks) * 0.08f + static_cast<float>(index);
        const float bob = std::sin(static_cast<float>(item.ageTicks) * 0.12f + index) * 0.05f;
        const float c = std::cos(phase);
        const float s = std::sin(phase);

        EntityModel model;
        model.model = matrix_identity_float4x4;
        model.model.columns[0] = simd_make_float4(c, 0, -s, 0);
        model.model.columns[2] = simd_make_float4(s, 0, c, 0);
        model.model.columns[3] = simd_make_float4(
            item.position.x, item.position.y + ItemEntity::SIZE * 0.5f + 0.1f + bob,
            item.position.z, 1.0f);
        [encoder setVertexBytes:&model length:sizeof(model) atIndex:2];

        [encoder setVertexBuffer:mesh.vertexBuffer offset:0 atIndex:0];
        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:mesh.indexCount
                             indexType:MTLIndexTypeUInt16
                           indexBuffer:mesh.indexBuffer
                     indexBufferOffset:0];
    }
}

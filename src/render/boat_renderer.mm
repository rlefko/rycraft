#include "render/boat_renderer.hpp"

#include "common/error.hpp"
#include "render/pixel_formats.hpp"

#include <cmath>
#include <vector>

namespace {

// Append the six faces of an axis-aligned box in boat-local space.
void appendBox(std::vector<EntityVertex>& vertices, std::vector<uint16_t>& indices, simd_float3 lo,
               simd_float3 hi, simd_float3 color) {
    struct Face {
        simd_float3 normal;
        simd_float3 corners[4];
    };
    const Face faces[6] = {
        {{1, 0, 0},
         {{hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z}, {hi.x, lo.y, hi.z}}},
        {{-1, 0, 0},
         {{lo.x, lo.y, hi.z}, {lo.x, hi.y, hi.z}, {lo.x, hi.y, lo.z}, {lo.x, lo.y, lo.z}}},
        {{0, 0, 1},
         {{hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}, {lo.x, lo.y, hi.z}}},
        {{0, 0, -1},
         {{lo.x, lo.y, lo.z}, {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, lo.y, lo.z}}},
        {{0, 1, 0},
         {{lo.x, hi.y, lo.z}, {lo.x, hi.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, hi.y, lo.z}}},
        {{0, -1, 0},
         {{lo.x, lo.y, hi.z}, {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z}}},
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

} // namespace

void BoatRenderer::buildMesh() {
    const simd_float3 wood = simd_make_float3(0.61f, 0.42f, 0.23f);
    const simd_float3 rim = simd_make_float3(0.70f, 0.49f, 0.28f);
    const float halfW = Boat::WIDTH * 0.5f;
    const float halfL = Boat::LENGTH * 0.5f;
    const float h = Boat::HEIGHT;
    const float wall = 0.18f; // hull thickness

    std::vector<EntityVertex> vertices;
    std::vector<uint16_t> indices;
    const float floorTop = h * 0.4f;
    // Floor slab, then four low gunwales, leaving the top open like a boat.
    appendBox(vertices, indices, simd_make_float3(-halfW, 0.f, -halfL),
              simd_make_float3(halfW, floorTop, halfL), wood);
    appendBox(vertices, indices, simd_make_float3(-halfW, floorTop, -halfL),
              simd_make_float3(halfW, h, -halfL + wall), rim);
    appendBox(vertices, indices, simd_make_float3(-halfW, floorTop, halfL - wall),
              simd_make_float3(halfW, h, halfL), rim);
    appendBox(vertices, indices, simd_make_float3(-halfW, floorTop, -halfL),
              simd_make_float3(-halfW + wall, h, halfL), rim);
    appendBox(vertices, indices, simd_make_float3(halfW - wall, floorTop, -halfL),
              simd_make_float3(halfW, h, halfL), rim);

    _indexCount = static_cast<uint32_t>(indices.size());
    _vertexBuffer = [_device newBufferWithBytes:vertices.data()
                                         length:vertices.size() * sizeof(EntityVertex)
                                        options:MTLResourceStorageModeShared];
    _indexBuffer = [_device newBufferWithBytes:indices.data()
                                        length:indices.size() * sizeof(uint16_t)
                                       options:MTLResourceStorageModeShared];
    if (!_vertexBuffer || !_indexBuffer) {
        RY_LOG_FATAL("Failed to allocate boat mesh buffers");
    }
}

BoatRenderer::BoatRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary) : _device(device) {
    id<MTLFunction> vertexFunc = [shaderLibrary newFunctionWithName:@"entityVertexMain"];
    id<MTLFunction> fragmentFunc = [shaderLibrary newFunctionWithName:@"entityFragmentMain"];
    if (!vertexFunc || !fragmentFunc) {
        RY_LOG_FATAL("Failed to load entity shader functions for boat rendering");
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
        RY_LOG_FATAL("Failed to create boat pipeline state");
    }
    buildMesh();
}

void BoatRenderer::render(id<MTLRenderCommandEncoder> encoder, id<MTLBuffer> uniformsBuffer,
                          uint64_t uniformsOffset, const std::vector<Boat>& boats,
                          const std::function<bool(const AABB&)>& isVisible) {
    if (boats.empty())
        return;

    [encoder setRenderPipelineState:_pipelineState];
    [encoder setVertexBuffer:uniformsBuffer offset:uniformsOffset atIndex:1];
    [encoder setFragmentBuffer:uniformsBuffer offset:uniformsOffset atIndex:1];
    [encoder setVertexBuffer:_vertexBuffer offset:0 atIndex:0];

    for (const Boat& boat : boats) {
        if (!isVisible(boat.getAABB()))
            continue;

        const float c = std::cos(boat.yaw);
        const float s = std::sin(boat.yaw);
        EntityModel model;
        model.model = matrix_identity_float4x4;
        model.model.columns[0] = simd_make_float4(c, 0, -s, 0);
        model.model.columns[2] = simd_make_float4(s, 0, c, 0);
        model.model.columns[3] =
            simd_make_float4(boat.position.x, boat.position.y, boat.position.z, 1.0f);
        [encoder setVertexBytes:&model length:sizeof(model) atIndex:2];

        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:_indexCount
                             indexType:MTLIndexTypeUInt16
                           indexBuffer:_indexBuffer
                     indexBufferOffset:0];
    }
}

#pragma once

#import <Metal/Metal.h>

#include "entity/boat.hpp"
#include "render/shader_types.hpp"

#include <functional>
#include <vector>

// ---------------------------------------------------------------------------
// BoatRenderer - draws rideable boats as small wooden hulls.
//
// Sibling of ItemEntityRenderer: it reuses the exact entity shader, vertex
// format, and MSAA scene pipeline, so boats receive the same sun/fog/shadow as
// terrain with no new GPU-shared struct. One hull mesh (a shallow open box)
// is baked once and drawn per boat with a yaw rotation from its heading.
// ---------------------------------------------------------------------------
class BoatRenderer {
public:
    BoatRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);

    void render(id<MTLRenderCommandEncoder> encoder, id<MTLBuffer> uniformsBuffer,
                uint64_t uniformsOffset, const std::vector<Boat>& boats,
                const std::function<bool(const AABB&)>& isVisible);

private:
    void buildMesh();

    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLBuffer> _vertexBuffer;
    id<MTLBuffer> _indexBuffer;
    uint32_t _indexCount = 0;
};

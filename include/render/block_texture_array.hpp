#pragma once

#import <Metal/Metal.h>

#include "render/block_textures.hpp"

#include <cstdint>

// ---------------------------------------------------------------------------
// BlockTextureArray — GPU array texture of procedural block textures.
//
// One 16×16 BGRA8 layer per TextureId entry (see block_textures.hpp), all
// generated and uploaded eagerly at construction. Greedy-merged quads sample
// it with a repeat-addressing sampler and UVs spanning the quad extent, so
// textures tile per block with no atlas bleed and no per-quad UV math.
// ---------------------------------------------------------------------------
class BlockTextureArray {
public:
    static constexpr uint32_t TILE_SIZE = 16;

    explicit BlockTextureArray(id<MTLDevice> device);

    id<MTLTexture> texture() const { return _texture; }
    id<MTLSamplerState> sampler() const { return _sampler; }

private:
    id<MTLTexture> _texture;
    id<MTLSamplerState> _sampler;

    // Paint one layer's pixels procedurally and upload it.
    void generateLayer(uint8_t layer);
};

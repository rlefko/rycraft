#pragma once

#include "render/vertex.hpp"
#include "world/chunk.hpp"

#include <cstdint>

// ---------------------------------------------------------------------------
// Block texture mapping — the single source of truth for which texture layer
// a block face samples.
//
// Block textures live in a 2D array texture (one 16×16 layer per entry, see
// BlockTextureArray). Layers 0..BlockType::COUNT-1 map 1:1 to block types;
// extra layers follow for per-face variants and UI needs. The mesher packs
// the layer into each vertex's faceAttr, so the fragment shader needs no
// lookup table.
// ---------------------------------------------------------------------------

// Extra layers beyond the per-block-type ones.
inline constexpr uint8_t TEXTURE_LAYER_GRASS_SIDE = static_cast<uint8_t>(BlockType::COUNT);
inline constexpr uint8_t TEXTURE_LAYER_WHITE = TEXTURE_LAYER_GRASS_SIDE + 1;
inline constexpr uint8_t TEXTURE_LAYER_COUNT = TEXTURE_LAYER_WHITE + 1;

// Which array layer a given face of a block samples.
constexpr uint8_t textureLayerFor(BlockType type, FaceNormal face) {
    if (type == BlockType::GRASS) {
        if (face == FaceNormal::PlusY) return static_cast<uint8_t>(BlockType::GRASS);
        if (face == FaceNormal::MinusY) return static_cast<uint8_t>(BlockType::DIRT);
        return TEXTURE_LAYER_GRASS_SIDE;
    }
    return static_cast<uint8_t>(type);
}

// Pack a face normal and texture layer into the vertex faceAttr field.
// The face fits in 3 bits (6 values); the layer rides above it.
constexpr uint32_t packFaceAttr(FaceNormal face, uint8_t layer) {
    return static_cast<uint32_t>(face) | (static_cast<uint32_t>(layer) << 3);
}

constexpr FaceNormal unpackFace(uint32_t faceAttr) {
    return static_cast<FaceNormal>(faceAttr & 7u);
}

constexpr uint8_t unpackTextureLayer(uint32_t faceAttr) {
    return static_cast<uint8_t>(faceAttr >> 3);
}

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

// Extra layers beyond the per-block-type ones (anchored at BlockType::COUNT
// so they slide automatically when the enum grows).
inline constexpr uint8_t TEXTURE_LAYER_GRASS_SIDE = static_cast<uint8_t>(BlockType::COUNT);
inline constexpr uint8_t TEXTURE_LAYER_WHITE = TEXTURE_LAYER_GRASS_SIDE + 1;
inline constexpr uint8_t TEXTURE_LAYER_LOG_TOP = TEXTURE_LAYER_WHITE + 1;
inline constexpr uint8_t TEXTURE_LAYER_BIRCH_LOG_TOP = TEXTURE_LAYER_LOG_TOP + 1;
inline constexpr uint8_t TEXTURE_LAYER_CACTUS_TOP = TEXTURE_LAYER_BIRCH_LOG_TOP + 1;
inline constexpr uint8_t TEXTURE_LAYER_SANDSTONE_TOP = TEXTURE_LAYER_CACTUS_TOP + 1;
inline constexpr uint8_t TEXTURE_LAYER_COUNT = TEXTURE_LAYER_SANDSTONE_TOP + 1;

// Which array layer a given face of a block samples.
constexpr uint8_t textureLayerFor(BlockType type, FaceNormal face) {
    switch (type) {
        case BlockType::GRASS:
            if (face == FaceNormal::PLUS_Y) return static_cast<uint8_t>(BlockType::GRASS);
            if (face == FaceNormal::MINUS_Y) return static_cast<uint8_t>(BlockType::DIRT);
            return TEXTURE_LAYER_GRASS_SIDE;
        case BlockType::LOG:
            if (face == FaceNormal::PLUS_Y || face == FaceNormal::MINUS_Y)
                return TEXTURE_LAYER_LOG_TOP;
            return static_cast<uint8_t>(type);
        case BlockType::BIRCH_LOG:
            if (face == FaceNormal::PLUS_Y || face == FaceNormal::MINUS_Y)
                return TEXTURE_LAYER_BIRCH_LOG_TOP;
            return static_cast<uint8_t>(type);
        case BlockType::SPRUCE_LOG: // dark bark shares the oak end-grain rings
            if (face == FaceNormal::PLUS_Y || face == FaceNormal::MINUS_Y)
                return TEXTURE_LAYER_LOG_TOP;
            return static_cast<uint8_t>(type);
        case BlockType::CACTUS:
            if (face == FaceNormal::PLUS_Y || face == FaceNormal::MINUS_Y)
                return TEXTURE_LAYER_CACTUS_TOP;
            return static_cast<uint8_t>(type);
        case BlockType::SANDSTONE:
            if (face == FaceNormal::PLUS_Y || face == FaceNormal::MINUS_Y)
                return TEXTURE_LAYER_SANDSTONE_TOP;
            return static_cast<uint8_t>(type);
        default:
            return static_cast<uint8_t>(type);
    }
}

// Pack a face normal, texture layer, and sky-light level into the vertex
// faceAttr field: face in bits 0-2, layer in bits 3-10, light in bits 11-14.
// Light 15 = open sky; lower values darken faces under cover (the column
// skylight that gives trees and overhangs their cast shadows).
constexpr uint32_t packFaceAttr(FaceNormal face, uint8_t layer, uint8_t skyLight = 15) {
    return static_cast<uint32_t>(face) | (static_cast<uint32_t>(layer) << 3) |
           (static_cast<uint32_t>(skyLight & 15u) << 11);
}

constexpr FaceNormal unpackFace(uint32_t faceAttr) {
    return static_cast<FaceNormal>(faceAttr & 7u);
}

constexpr uint8_t unpackTextureLayer(uint32_t faceAttr) {
    return static_cast<uint8_t>((faceAttr >> 3) & 0xFFu);
}

constexpr uint8_t unpackSkyLight(uint32_t faceAttr) {
    return static_cast<uint8_t>((faceAttr >> 11) & 15u);
}

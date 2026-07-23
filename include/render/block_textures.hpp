#pragma once

#include "render/vertex.hpp"
#include "world/chunk.hpp"
#include "world/item.hpp"

#include <cstdint>

// ---------------------------------------------------------------------------
// Block texture mapping, the single source of truth for which texture layer
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
inline constexpr uint8_t TEXTURE_LAYER_CRAFTING_TABLE_TOP = TEXTURE_LAYER_SANDSTONE_TOP + 1;
inline constexpr uint8_t TEXTURE_LAYER_FURNACE_TOP = TEXTURE_LAYER_CRAFTING_TABLE_TOP + 1;
inline constexpr uint8_t TEXTURE_LAYER_FURNACE_SIDE = TEXTURE_LAYER_FURNACE_TOP + 1;
inline constexpr uint8_t TEXTURE_LAYER_CHEST_SIDE = TEXTURE_LAYER_FURNACE_SIDE + 1;
inline constexpr uint8_t TEXTURE_LAYER_CHEST_TOP = TEXTURE_LAYER_CHEST_SIDE + 1;
inline constexpr uint8_t TEXTURE_LAYER_COUNT = TEXTURE_LAYER_CHEST_TOP + 1;

// UI item-icon layers append after every block-face layer. Only the overlay
// samples them; faceAttr packing never sees these indices. The non-block
// item range is contiguous, so a layer is a constexpr offset.
inline constexpr uint8_t TEXTURE_LAYER_ITEM_FIRST = TEXTURE_LAYER_COUNT;
inline constexpr size_t ITEM_ICON_COUNT = NON_BLOCK_ITEM_COUNT;
inline constexpr uint16_t TEXTURE_LAYER_TOTAL =
    static_cast<uint16_t>(TEXTURE_LAYER_ITEM_FIRST) + static_cast<uint16_t>(ITEM_ICON_COUNT);
static_assert(TEXTURE_LAYER_TOTAL <= 255, "texture array layers must stay 8-bit addressable");

constexpr uint8_t itemIconLayer(ItemType type) {
    return static_cast<uint8_t>(TEXTURE_LAYER_ITEM_FIRST +
                                (static_cast<uint16_t>(type) - ITEM_ID_BASE));
}

// Per-texel emission is intentionally narrower than the block-level emissive
// bit carried by faceAttr. The bit remains a fast shader gate, while this mask
// keeps a lit furnace's stone shell and a torch's wooden stick from glowing.
constexpr uint8_t emissionMaskForTexel(uint8_t layer, uint8_t x, uint8_t y) {
    if (x >= 16 || y >= 16) return 0;
    if (layer == static_cast<uint8_t>(BlockType::LAVA)) return 255;
    if (layer == static_cast<uint8_t>(BlockType::FURNACE_LIT)) {
        return x >= 3 && x < 13 && y >= 6 && y < 13 ? 255 : 0;
    }
    if (layer == static_cast<uint8_t>(BlockType::TORCH)) {
        return x >= 6 && x < 10 && y >= 1 && y < 5 ? 255 : 0;
    }
    return 0;
}

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
        case BlockType::ACACIA_LOG:
        case BlockType::JUNGLE_LOG:
        case BlockType::MANGROVE_LOG:
        case BlockType::PALM_LOG:
        case BlockType::WILLOW_LOG:
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
        case BlockType::CRAFTING_TABLE:
            if (face == FaceNormal::PLUS_Y) return TEXTURE_LAYER_CRAFTING_TABLE_TOP;
            if (face == FaceNormal::MINUS_Y) return static_cast<uint8_t>(BlockType::PLANKS);
            return static_cast<uint8_t>(type);
        case BlockType::FURNACE:
        case BlockType::FURNACE_LIT:
            if (face == FaceNormal::PLUS_Y || face == FaceNormal::MINUS_Y)
                return TEXTURE_LAYER_FURNACE_TOP;
            // Saves do not yet carry facing bits. The canonical fixed front
            // faces world -Z, and only that face owns the furnace mouth.
            return face == FaceNormal::MINUS_Z ? static_cast<uint8_t>(type)
                                               : TEXTURE_LAYER_FURNACE_SIDE;
        case BlockType::CHEST:
            if (face == FaceNormal::PLUS_Y || face == FaceNormal::MINUS_Y)
                return TEXTURE_LAYER_CHEST_TOP;
            // Chests share the same fixed -Z front until facing is persisted.
            return face == FaceNormal::MINUS_Z ? static_cast<uint8_t>(type)
                                               : TEXTURE_LAYER_CHEST_SIDE;
        default:
            return static_cast<uint8_t>(type);
    }
}

// The isometric inventory icon may rotate a fixed-front block toward the
// viewer without changing its world orientation. Other cubes retain the
// ordinary +Z right face.
constexpr FaceNormal itemIconRightFaceFor(BlockType type) {
    return type == BlockType::FURNACE || type == BlockType::FURNACE_LIT || type == BlockType::CHEST
               ? FaceNormal::MINUS_Z
               : FaceNormal::PLUS_Z;
}

// Pack the per-vertex face attributes into the faceAttr field: face in bits
// 0-2, layer in bits 3-10, sky light in bits 11-14, corner AO in bits 15-16,
// block light in bits 17-20, emissive flag in bit 21, sway class in bits
// 22-23. Sky light 15 = open sky; lower values darken faces under cover.
// Corner AO 3 = fully open, 0 = a fully enclosed voxel corner. Block light
// 0-15 is propagated emitter light reaching a face; emissive marks a source
// surface that ignores sun, shadow, and sky. Sway (see swayClass) picks the
// wind animation the scene and shadow vertex stages both apply.
constexpr uint32_t packFaceAttr(FaceNormal face, uint8_t layer, uint8_t skyLight = 15,
                                uint8_t cornerAO = 3, uint8_t blockLight = 0, bool emissive = false,
                                uint8_t sway = 0) {
    return static_cast<uint32_t>(face) | (static_cast<uint32_t>(layer) << 3) |
           (static_cast<uint32_t>(skyLight & 15u) << 11) |
           (static_cast<uint32_t>(cornerAO & 3u) << 15) |
           (static_cast<uint32_t>(blockLight & 15u) << 17) |
           (static_cast<uint32_t>(emissive ? 1u : 0u) << 21) |
           (static_cast<uint32_t>(sway & 3u) << 22);
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

constexpr uint8_t unpackCornerAO(uint32_t faceAttr) {
    return static_cast<uint8_t>((faceAttr >> 15) & 3u);
}

constexpr uint8_t unpackBlockLight(uint32_t faceAttr) {
    return static_cast<uint8_t>((faceAttr >> 17) & 15u);
}

constexpr bool unpackEmissive(uint32_t faceAttr) {
    return ((faceAttr >> 21) & 1u) != 0u;
}

constexpr uint8_t unpackSway(uint32_t faceAttr) {
    return static_cast<uint8_t>((faceAttr >> 22) & 3u);
}

// Water uses high bits without changing the 16-byte vertex layout. Values 0
// through 4 encode still, west, east, north, and south flow in bits 24-26.
// Bit 27 identifies a falling column. Bit 30 records independent exterior
// sky authority for water reflection and shadow fallback. Bits 0-23 retain
// the shared face, texture, sky, AO, block-light, emissive, and sway semantics.
inline constexpr uint32_t FLUID_EXTERIOR_SKY_ATTRIBUTE_MASK = 1U << 30U;

constexpr uint32_t packFluidFaceAttr(FaceNormal face, uint8_t skyLight, uint8_t flowDirection,
                                     bool falling, uint8_t blockLight = 0,
                                     bool exteriorSky = true) {
    return packFaceAttr(face, static_cast<uint8_t>(BlockType::WATER), skyLight, 3, blockLight) |
           (static_cast<uint32_t>(flowDirection & 7u) << 24) |
           (static_cast<uint32_t>(falling) << 27) |
           (exteriorSky ? FLUID_EXTERIOR_SKY_ATTRIBUTE_MASK : 0U);
}

constexpr uint8_t unpackFluidDirection(uint32_t faceAttr) {
    return static_cast<uint8_t>((faceAttr >> 24) & 7u);
}

constexpr bool unpackFluidFalling(uint32_t faceAttr) {
    return ((faceAttr >> 27) & 1u) != 0;
}

constexpr bool unpackFluidExteriorSky(uint32_t faceAttr) {
    return (faceAttr & FLUID_EXTERIOR_SKY_ATTRIBUTE_MASK) != 0U;
}

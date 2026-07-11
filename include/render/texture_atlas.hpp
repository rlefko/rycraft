#pragma once

#import <Metal/Metal.h>
#include <array>
#include <cstdint>
#include <mutex>

class TextureAtlas {
public:
    static constexpr uint32_t ATLAS_WIDTH = 1024;
    static constexpr uint32_t ATLAS_HEIGHT = 1024;
    static constexpr uint32_t TILE_SIZE = 16;
    static constexpr uint32_t TILES_PER_ROW = ATLAS_WIDTH / TILE_SIZE;
    static constexpr uint32_t TOTAL_TILES = TILES_PER_ROW * (ATLAS_HEIGHT / TILE_SIZE);
    static constexpr uint32_t MAX_BLOCK_TYPES = 256;

    struct TileInfo {
        float u;
        float v;
        float uSize;
        float vSize;
    };

    explicit TextureAtlas(id<MTLDevice> device);

    // Allocate a tile for a block type, returns UV coords
    TileInfo allocate(uint32_t blockType);

    // Get UV for a block type (allocates if not yet cached)
    TileInfo getUV(uint32_t blockType);

    // Get the MTLTexture
    id<MTLTexture> texture() const;

    // Get the sampler state (nearest-filter)
    id<MTLSamplerState> sampler() const;

private:
    id<MTLTexture> _texture;
    id<MTLSamplerState> _sampler;
    std::array<bool, TOTAL_TILES> _allocated;
    std::array<TileInfo, MAX_BLOCK_TYPES> _tileCache;
    std::array<bool, MAX_BLOCK_TYPES> _cacheValid;
    uint32_t _nextTile;
    std::mutex _mutex;

    void generateBlockTexture(uint32_t blockType, uint32_t tileIndex);
    TileInfo computeTileUV(uint32_t tileIndex) const;
};

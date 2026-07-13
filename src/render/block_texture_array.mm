#include "render/block_texture_array.hpp"

#include "common/error.hpp"
#include "world/chunk.hpp"
#include "world/noise.hpp"

#include <array>
#include <cmath>
#include <cstring>

// BGRA8Unorm pixel — 4 bytes per pixel
struct alignas(4) BgraPixel {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
};

static uint8_t clampToByte(double value) {
    if (value <= 0.0)
        return 0;
    if (value >= 1.0)
        return 255;
    return static_cast<uint8_t>(std::lround(value * 255.0));
}

static void fillTilePixel(BgraPixel* pixel, double baseR, double baseG, double baseB,
                          double noiseValue, double noiseAmp = 0.1, uint8_t alpha = 255) {
    double factor = 1.0 + noiseValue * noiseAmp;
    pixel->r = clampToByte(baseR * factor);
    pixel->g = clampToByte(baseG * factor);
    pixel->b = clampToByte(baseB * factor);
    pixel->a = alpha;
}

void BlockTextureArray::generateLayer(uint8_t layer) {
    std::array<BgraPixel, TILE_SIZE * TILE_SIZE> tilePixels;
    std::memset(tilePixels.data(), 0, tilePixels.size() * sizeof(BgraPixel));

    SimplexNoise noise(static_cast<uint32_t>(layer) * 7919 + 104729);

    auto getTilePixel = [&](uint32_t px, uint32_t py) -> BgraPixel& {
        return tilePixels[py * TILE_SIZE + px];
    };

    // Painters for the non-block layers first, block types below.
    if (layer == TEXTURE_LAYER_WHITE) {
        for (auto& p : tilePixels) {
            p.r = p.g = p.b = p.a = 255;
        }
    } else if (layer == TEXTURE_LAYER_GRASS_SIDE) {
        // Dirt base with a strip of grass hanging over the top edge
        for (uint32_t y = 0; y < TILE_SIZE; ++y) {
            for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                double n = noise.noise2D(x * 0.2, y * 0.2);
                bool grassStrip = y < 3 || (y == 3 && noise.noise2D(x * 0.9, 7.0) > 0.0);
                if (grassStrip) {
                    fillTilePixel(&getTilePixel(x, y), 0.3, 0.6, 0.2, n, 0.12);
                } else {
                    fillTilePixel(&getTilePixel(x, y), 0.4, 0.25, 0.15, n, 0.12);
                }
            }
        }
    } else {
        switch (static_cast<BlockType>(layer)) {
            case BlockType::STONE: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.15, y * 0.15);
                        fillTilePixel(&getTilePixel(x, y), 0.5, 0.5, 0.5, n, 0.15);
                    }
                }
                // Crack lines
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.5 + 100, y * 0.5 + 100);
                        if (n > 0.7) {
                            auto& p = getTilePixel(x, y);
                            p.r = p.g = p.b = clampToByte(0.35);
                        }
                    }
                }
                break;
            }

            case BlockType::DIRT: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.2, y * 0.2);
                        fillTilePixel(&getTilePixel(x, y), 0.4, 0.25, 0.15, n, 0.12);
                    }
                }
                break;
            }

            case BlockType::GRASS: {
                // The grass top face — sides use TEXTURE_LAYER_GRASS_SIDE
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.2, y * 0.2);
                        fillTilePixel(&getTilePixel(x, y), 0.3, 0.6, 0.2, n, 0.12);
                    }
                }
                break;
            }

            case BlockType::SAND: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.3, y * 0.3);
                        fillTilePixel(&getTilePixel(x, y), 0.8, 0.75, 0.5, n, 0.06);
                    }
                }
                break;
            }

            case BlockType::WATER: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double wave = noise.noise2D(x * 0.25 + y * 0.1, y * 0.25);
                        fillTilePixel(&getTilePixel(x, y), 0.2, 0.4, 0.8, wave, 0.1, 128);
                    }
                }
                break;
            }

            case BlockType::LOG: {
                uint32_t cx = TILE_SIZE / 2;
                uint32_t cy = TILE_SIZE / 2;
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double dx = static_cast<double>(x) - static_cast<double>(cx);
                        double dy = static_cast<double>(y) - static_cast<double>(cy);
                        double dist = std::sqrt(dx * dx + dy * dy);
                        double angle = std::atan2(dy, dx);
                        double ring = std::sin(dist * 1.2 + angle * 0.5) * 0.5 + 0.5;
                        double n = noise.noise2D(x * 0.3, y * 0.3) * 0.08;
                        double base = 0.35 + ring * 0.15 + n;
                        auto& p = getTilePixel(x, y);
                        p.r = clampToByte(base * 0.85);
                        p.g = clampToByte(base * 0.65);
                        p.b = clampToByte(base * 0.35);
                        p.a = 255;
                    }
                }
                break;
            }

            case BlockType::LEAVES: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.25, y * 0.25);
                        // Alpha-cutout foliage: sparse holes let the sky and
                        // the branches behind show through (the fragment
                        // shader discards texels below 0.5 alpha)
                        double holes = noise.noise2D(x * 0.7 + 31.0, y * 0.7 + 47.0);
                        if (holes > 0.45) {
                            getTilePixel(x, y).a = 0;
                        } else {
                            fillTilePixel(&getTilePixel(x, y), 0.2, 0.5, 0.15, n, 0.25);
                        }
                    }
                }
                break;
            }

            case BlockType::SNOW: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.2, y * 0.2);
                        fillTilePixel(&getTilePixel(x, y), 0.95, 0.95, 0.97, n, 0.04);
                    }
                }
                break;
            }

            case BlockType::PLANKS: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.15, y * 0.15);
                        bool isLine = (y % 4 == 0);
                        double baseR = isLine ? 0.5 : 0.6;
                        double baseG = isLine ? 0.35 : 0.45;
                        double baseB = isLine ? 0.2 : 0.28;
                        fillTilePixel(&getTilePixel(x, y), baseR, baseG, baseB, n, 0.06);
                    }
                }
                break;
            }

            case BlockType::BEDROCK: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.25, y * 0.25);
                        fillTilePixel(&getTilePixel(x, y), 0.3, 0.3, 0.3, n, 0.25);
                    }
                }
                break;
            }

            case BlockType::COAL_ORE:
            case BlockType::IRON_ORE:
            case BlockType::GOLD_ORE:
            case BlockType::DIAMOND_ORE: {
                // Stone base
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.15, y * 0.15);
                        fillTilePixel(&getTilePixel(x, y), 0.5, 0.5, 0.5, n, 0.12);
                    }
                }
                // Ore speckle
                double oreR = 0.0, oreG = 0.0, oreB = 0.0;
                switch (static_cast<BlockType>(layer)) {
                    case BlockType::COAL_ORE:
                        oreR = oreG = oreB = 0.1;
                        break;
                    case BlockType::IRON_ORE:
                        oreR = 0.8;
                        oreG = 0.6;
                        oreB = 0.45;
                        break;
                    case BlockType::GOLD_ORE:
                        oreR = 1.0;
                        oreG = 0.85;
                        oreB = 0.1;
                        break;
                    case BlockType::DIAMOND_ORE:
                        oreR = 0.4;
                        oreG = 0.9;
                        oreB = 0.9;
                        break;
                    default:
                        break;
                }
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.4 + 50, y * 0.4 + 50);
                        if (n > 0.55) {
                            auto& p = getTilePixel(x, y);
                            p.r = clampToByte(oreR);
                            p.g = clampToByte(oreG);
                            p.b = clampToByte(oreB);
                        }
                    }
                }
                break;
            }

            case BlockType::GRAVEL: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.25, y * 0.25);
                        fillTilePixel(&getTilePixel(x, y), 0.55, 0.55, 0.55, n, 0.15);
                    }
                }
                break;
            }

            case BlockType::GLASS: {
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        auto& p = getTilePixel(x, y);
                        // Subtle glass: light blue with high alpha, slight border
                        bool isEdge = (x == 0 || x == 15 || y == 0 || y == 15);
                        if (isEdge) {
                            p.r = 180;
                            p.g = 200;
                            p.b = 220;
                            p.a = 200;
                        } else {
                            p.r = 220;
                            p.g = 230;
                            p.b = 240;
                            p.a = 100;
                        }
                    }
                }
                break;
            }

            // AIR (never drawn) gets a transparent black layer
            default:
                break;
        }
    }

    [_texture replaceRegion:MTLRegionMake2D(0, 0, TILE_SIZE, TILE_SIZE)
                mipmapLevel:0
                      slice:layer
                  withBytes:tilePixels.data()
                bytesPerRow:TILE_SIZE * sizeof(BgraPixel)
              bytesPerImage:0];
}

BlockTextureArray::BlockTextureArray(id<MTLDevice> device) {
    auto descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = MTLTextureType2DArray;
    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.width = TILE_SIZE;
    descriptor.height = TILE_SIZE;
    descriptor.arrayLength = TEXTURE_LAYER_COUNT;
    descriptor.usage = MTLTextureUsageShaderRead;

    _texture = [device newTextureWithDescriptor:descriptor];
    if (!_texture) {
        RY_LOG_FATAL("Failed to allocate block texture array");
    }

    // Repeat addressing is what lets a single greedy quad tile its texture
    // across every block it covers.
    auto samplerDesc = [[MTLSamplerDescriptor alloc] init];
    samplerDesc.magFilter = MTLSamplerMinMagFilterNearest;
    samplerDesc.minFilter = MTLSamplerMinMagFilterNearest;
    samplerDesc.mipFilter = MTLSamplerMipFilterNotMipmapped;
    samplerDesc.sAddressMode = MTLSamplerAddressModeRepeat;
    samplerDesc.tAddressMode = MTLSamplerAddressModeRepeat;
    _sampler = [device newSamplerStateWithDescriptor:samplerDesc];
    if (!_sampler) {
        RY_LOG_FATAL("Failed to create block texture sampler");
    }

    for (uint8_t layer = 0; layer < TEXTURE_LAYER_COUNT; ++layer) {
        generateLayer(layer);
    }
}

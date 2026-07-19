#include "render/block_texture_array.hpp"

#include "common/error.hpp"
#include "world/chunk.hpp"
#include "world/noise.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

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

namespace {

constexpr uint8_t ALPHA_TEST_CUTOFF = 128;

std::vector<BgraPixel> downsampleAlphaAware(const std::vector<BgraPixel>& source,
                                            uint32_t sourceEdge) {
    const uint32_t destinationEdge = sourceEdge / 2;
    std::vector<BgraPixel> destination(destinationEdge * destinationEdge);

    for (uint32_t y = 0; y < destinationEdge; ++y) {
        for (uint32_t x = 0; x < destinationEdge; ++x) {
            uint32_t alphaSum = 0;
            uint32_t bluePremultiplied = 0;
            uint32_t greenPremultiplied = 0;
            uint32_t redPremultiplied = 0;
            for (uint32_t childY = 0; childY < 2; ++childY) {
                for (uint32_t childX = 0; childX < 2; ++childX) {
                    const BgraPixel& child = source[(y * 2 + childY) * sourceEdge + x * 2 + childX];
                    alphaSum += child.a;
                    bluePremultiplied += static_cast<uint32_t>(child.b) * child.a;
                    greenPremultiplied += static_cast<uint32_t>(child.g) * child.a;
                    redPremultiplied += static_cast<uint32_t>(child.r) * child.a;
                }
            }

            BgraPixel& result = destination[y * destinationEdge + x];
            result.a = static_cast<uint8_t>((alphaSum + 2) / 4);
            if (alphaSum == 0)
                continue;

            result.b = static_cast<uint8_t>((bluePremultiplied + alphaSum / 2) / alphaSum);
            result.g = static_cast<uint8_t>((greenPremultiplied + alphaSum / 2) / alphaSum);
            result.r = static_cast<uint8_t>((redPremultiplied + alphaSum / 2) / alphaSum);
        }
    }
    return destination;
}

uint32_t alphaCoverage(const std::vector<BgraPixel>& pixels) {
    return static_cast<uint32_t>(std::count_if(pixels.begin(), pixels.end(), [](BgraPixel pixel) {
        return pixel.a >= ALPHA_TEST_CUTOFF;
    }));
}

void preserveAlphaCoverage(std::vector<BgraPixel>& pixels, uint32_t baseCovered,
                           uint32_t baseTexelCount) {
    if (baseCovered == 0 || baseCovered == baseTexelCount)
        return;

    // Quantize the base-level coverage to this mip's texel count. A nonempty
    // cutout keeps at least one covered texel so thin flora cannot disappear
    // completely at the tail of the mip chain.
    uint32_t desiredCovered = static_cast<uint32_t>(
        (static_cast<uint64_t>(baseCovered) * pixels.size() + baseTexelCount / 2) / baseTexelCount);
    desiredCovered = std::clamp(desiredCovered, 1U, static_cast<uint32_t>(pixels.size()));

    // Rank by the box-filtered alpha and resolve ties by texel index. Moving
    // the selected values to opposite sides of the shader's fixed cutoff
    // preserves the exact representable coverage without platform-dependent
    // floating-point searches or black fringes in the RGB channels.
    std::vector<uint32_t> ranked(pixels.size());
    std::iota(ranked.begin(), ranked.end(), 0U);
    std::stable_sort(ranked.begin(), ranked.end(), [&](uint32_t left, uint32_t right) {
        return pixels[left].a > pixels[right].a;
    });
    for (uint32_t rank = 0; rank < ranked.size(); ++rank) {
        BgraPixel& pixel = pixels[ranked[rank]];
        if (rank < desiredCovered) {
            pixel.a = std::max(pixel.a, ALPHA_TEST_CUTOFF);
        } else {
            pixel.a = std::min<uint8_t>(pixel.a, ALPHA_TEST_CUTOFF - 1);
        }
    }
}

} // namespace

static void uploadLayerMips(id<MTLTexture> texture,
                            const std::array<BgraPixel, BlockTextureArray::TILE_SIZE *
                                                            BlockTextureArray::TILE_SIZE>& pixels,
                            uint8_t layer);
static void paintItemIcon(BgraPixel* pixels, ItemType item, SimplexNoise& noise);

void BlockTextureArray::generateLayer(uint8_t layer) {
    std::array<BgraPixel, TILE_SIZE * TILE_SIZE> tilePixels;
    std::memset(tilePixels.data(), 0, tilePixels.size() * sizeof(BgraPixel));

    SimplexNoise noise(static_cast<uint32_t>(layer) * 7919 + 104729);

    auto getTilePixel = [&](uint32_t px, uint32_t py) -> BgraPixel& {
        return tilePixels[py * TILE_SIZE + px];
    };

    // Item-icon layers: simple procedural pixel art on a transparent tile,
    // shaded with the same noise the block layers use.
    if (layer >= TEXTURE_LAYER_ITEM_FIRST) {
        const auto item = static_cast<ItemType>(ITEM_ID_BASE + (layer - TEXTURE_LAYER_ITEM_FIRST));
        paintItemIcon(tilePixels.data(), item, noise);
        uploadLayerMips(_texture, tilePixels, layer);
        return;
    }

    // Ring cross-section shared by the log end-grain layers.
    auto paintLogRings = [&](double palette) {
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
                double base = (0.35 + ring * 0.15 + n) * palette;
                auto& p = getTilePixel(x, y);
                p.r = clampToByte(base * 0.85);
                p.g = clampToByte(base * 0.65);
                p.b = clampToByte(base * 0.35);
                p.a = 255;
            }
        }
    };

    // Vertical bark streaks shared by the log side layers.
    auto paintBark = [&](double r, double g, double b, double streakDarken) {
        for (uint32_t y = 0; y < TILE_SIZE; ++y) {
            for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                // Stretch the noise along y so the grain runs vertically
                double n = noise.noise2D(x * 0.6, y * 0.12);
                double streak = noise.noise2D(x * 0.9 + 40.0, y * 0.08);
                double f = streak > 0.35 ? 1.0 - streakDarken : 1.0;
                fillTilePixel(&getTilePixel(x, y), r * f, g * f, b * f, n, 0.12);
            }
        }
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
    } else if (layer == TEXTURE_LAYER_LOG_TOP) {
        paintLogRings(1.0);
    } else if (layer == TEXTURE_LAYER_BIRCH_LOG_TOP) {
        paintLogRings(1.35);
    } else if (layer == TEXTURE_LAYER_CACTUS_TOP) {
        for (uint32_t y = 0; y < TILE_SIZE; ++y) {
            for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                double dx = static_cast<double>(x) - 7.5;
                double dy = static_cast<double>(y) - 7.5;
                double dist = std::sqrt(dx * dx + dy * dy);
                double n = noise.noise2D(x * 0.3, y * 0.3);
                double lighten = dist < 4.0 ? 1.25 : 1.0;
                fillTilePixel(&getTilePixel(x, y), 0.2 * lighten, 0.45 * lighten, 0.15 * lighten, n,
                              0.1);
            }
        }
    } else if (layer == TEXTURE_LAYER_SANDSTONE_TOP) {
        for (uint32_t y = 0; y < TILE_SIZE; ++y) {
            for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                double n = noise.noise2D(x * 0.25, y * 0.25);
                fillTilePixel(&getTilePixel(x, y), 0.82, 0.74, 0.52, n, 0.05);
            }
        }
    } else if (layer == TEXTURE_LAYER_CRAFTING_TABLE_TOP) {
        // Plank base scored by a dark 2x2 work-grid
        for (uint32_t y = 0; y < TILE_SIZE; ++y) {
            for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                double n = noise.noise2D(x * 0.15, y * 0.15);
                bool grid = (x == 0 || x == 7 || x == 8 || x == 15 || y == 0 || y == 7 || y == 8 ||
                             y == 15);
                double f = grid ? 0.55 : 1.0;
                fillTilePixel(&getTilePixel(x, y), 0.6 * f, 0.45 * f, 0.28 * f, n, 0.06);
            }
        }
    } else if (layer == TEXTURE_LAYER_FURNACE_TOP) {
        for (uint32_t y = 0; y < TILE_SIZE; ++y) {
            for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                double n = noise.noise2D(x * 0.3, y * 0.3);
                fillTilePixel(&getTilePixel(x, y), 0.42, 0.42, 0.44, n, 0.12);
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
                // Side faces: vertical bark (the end-grain rings live in
                // TEXTURE_LAYER_LOG_TOP)
                paintBark(0.45, 0.33, 0.18, 0.35);
                break;
            }

            case BlockType::ACACIA_LOG:
                paintBark(0.55, 0.28, 0.12, 0.32);
                break;
            case BlockType::JUNGLE_LOG:
                paintBark(0.42, 0.27, 0.14, 0.30);
                break;
            case BlockType::MANGROVE_LOG:
                paintBark(0.38, 0.18, 0.12, 0.38);
                break;
            case BlockType::PALM_LOG:
                paintBark(0.58, 0.43, 0.22, 0.24);
                break;
            case BlockType::WILLOW_LOG:
                paintBark(0.35, 0.31, 0.19, 0.28);
                break;

            case BlockType::BIRCH_LOG: {
                // Near-white bark with dark horizontal dash patches
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.4, y * 0.15);
                        double dash = noise.noise2D(x * 0.25 + 60.0, y * 1.1 + 60.0);
                        if (dash > 0.62) {
                            fillTilePixel(&getTilePixel(x, y), 0.2, 0.18, 0.14, n, 0.1);
                        } else {
                            fillTilePixel(&getTilePixel(x, y), 0.85, 0.84, 0.78, n, 0.05);
                        }
                    }
                }
                break;
            }

            case BlockType::SPRUCE_LOG: {
                paintBark(0.3, 0.2, 0.1, 0.4);
                break;
            }

            case BlockType::LEAVES:
            case BlockType::BIRCH_LEAVES:
            case BlockType::SPRUCE_LEAVES:
            case BlockType::ACACIA_LEAVES:
            case BlockType::JUNGLE_LEAVES:
            case BlockType::MANGROVE_LEAVES:
            case BlockType::PALM_LEAVES:
            case BlockType::WILLOW_LEAVES: {
                double leafR = 0.2, leafG = 0.5, leafB = 0.15; // oak
                BlockType leaf = static_cast<BlockType>(layer);
                if (leaf == BlockType::BIRCH_LEAVES) {
                    leafR = 0.35;
                    leafG = 0.55;
                    leafB = 0.25;
                } else if (leaf == BlockType::SPRUCE_LEAVES) {
                    leafR = 0.12;
                    leafG = 0.35;
                    leafB = 0.18;
                } else if (leaf == BlockType::ACACIA_LEAVES) {
                    leafR = 0.30;
                    leafG = 0.52;
                    leafB = 0.16;
                } else if (leaf == BlockType::JUNGLE_LEAVES) {
                    leafR = 0.10;
                    leafG = 0.48;
                    leafB = 0.13;
                } else if (leaf == BlockType::MANGROVE_LEAVES) {
                    leafR = 0.12;
                    leafG = 0.40;
                    leafB = 0.20;
                } else if (leaf == BlockType::PALM_LEAVES) {
                    leafR = 0.20;
                    leafG = 0.58;
                    leafB = 0.18;
                } else if (leaf == BlockType::WILLOW_LEAVES) {
                    leafR = 0.28;
                    leafG = 0.52;
                    leafB = 0.22;
                }
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
                            fillTilePixel(&getTilePixel(x, y), leafR, leafG, leafB, n, 0.25);
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

            case BlockType::MUD:
            case BlockType::CLAY:
            case BlockType::SILT:
            case BlockType::BASALT:
            case BlockType::VOLCANIC_ASH:
            case BlockType::LIMESTONE:
            case BlockType::OBSIDIAN:
            case BlockType::ANDESITE: {
                double r = 0.30, g = 0.23, b = 0.17;
                switch (static_cast<BlockType>(layer)) {
                    case BlockType::CLAY:
                        r = 0.56;
                        g = 0.58;
                        b = 0.60;
                        break;
                    case BlockType::SILT:
                        r = 0.50;
                        g = 0.43;
                        b = 0.31;
                        break;
                    case BlockType::BASALT:
                        r = 0.20;
                        g = 0.21;
                        b = 0.22;
                        break;
                    case BlockType::VOLCANIC_ASH:
                        r = 0.27;
                        g = 0.26;
                        b = 0.25;
                        break;
                    case BlockType::LIMESTONE:
                        r = 0.72;
                        g = 0.70;
                        b = 0.62;
                        break;
                    case BlockType::OBSIDIAN:
                        r = 0.12;
                        g = 0.08;
                        b = 0.17;
                        break;
                    case BlockType::ANDESITE:
                        r = 0.42;
                        g = 0.43;
                        b = 0.42;
                        break;
                    default:
                        break;
                }
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.24, y * 0.24);
                        double variation = 0.13;
                        switch (static_cast<BlockType>(layer)) {
                            case BlockType::BASALT:
                                n = n * 0.72 + noise.noise2D(x * 0.62, y * 0.18) * 0.28;
                                variation = 0.10;
                                break;
                            case BlockType::VOLCANIC_ASH:
                                n = noise.noise2D(x * 0.52, y * 0.52);
                                variation = 0.07;
                                break;
                            case BlockType::LIMESTONE:
                                n = n * 0.78 + ((y + 1) % 5 == 0 ? -0.22 : 0.0);
                                variation = 0.11;
                                break;
                            case BlockType::OBSIDIAN:
                                n = n * 0.45 + ((x + y * 2) % 13 == 0 ? 0.48 : -0.05);
                                variation = 0.09;
                                break;
                            case BlockType::ANDESITE:
                                n = n * 0.60 + noise.noise2D(x * 0.58, y * 0.58) * 0.40;
                                variation = 0.14;
                                break;
                            default:
                                break;
                        }
                        fillTilePixel(&getTilePixel(x, y), r, g, b, n, variation);
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

            case BlockType::COBBLESTONE:
            case BlockType::MOSSY_COBBLESTONE: {
                // Rounded stones: darken along cell borders of a 4px grid,
                // jittered by noise so the joints wander
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.3, y * 0.3);
                        double jitter = noise.noise2D(x * 0.8 + 20.0, y * 0.8 + 20.0);
                        bool joint = ((x + static_cast<uint32_t>(jitter * 2.0 + 2.0)) % 5 == 0) ||
                                     ((y + static_cast<uint32_t>(jitter * 2.0 + 2.0)) % 5 == 0);
                        double f = joint ? 0.6 : 1.0;
                        fillTilePixel(&getTilePixel(x, y), 0.45 * f, 0.45 * f, 0.47 * f, n, 0.15);
                    }
                }
                if (static_cast<BlockType>(layer) == BlockType::MOSSY_COBBLESTONE) {
                    for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                        for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                            double moss = noise.noise2D(x * 0.35 + 90.0, y * 0.35 + 90.0);
                            if (moss > 0.25) {
                                auto& p = getTilePixel(x, y);
                                p.r = clampToByte(0.25);
                                p.g = clampToByte(0.45);
                                p.b = clampToByte(0.2);
                            }
                        }
                    }
                }
                break;
            }

            case BlockType::SANDSTONE: {
                // Sand palette with horizontal strata every few rows
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.25, y * 0.25);
                        bool band = (y % 5 == 0);
                        double f = band ? 0.85 : 1.0;
                        fillTilePixel(&getTilePixel(x, y), 0.8 * f, 0.72 * f, 0.5 * f, n, 0.05);
                    }
                }
                break;
            }

            case BlockType::CACTUS: {
                // Green with darker vertical ridges
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.3, y * 0.15);
                        bool ridge = (x % 4 == 0);
                        double f = ridge ? 0.7 : 1.0;
                        fillTilePixel(&getTilePixel(x, y), 0.2 * f, 0.45 * f, 0.15 * f, n, 0.1);
                    }
                }
                break;
            }

            case BlockType::DEAD_BUSH: {
                // Sparse branching diagonal twigs from a bottom-center root;
                // background stays transparent (alpha-cutout cross quad)
                for (int i = 0; i < 4; ++i) {
                    double drift = (i - 1.5) * 0.9;
                    double px = 7.5;
                    for (int py = 15; py >= 4; --py) {
                        px += drift * 0.22 + noise.noise2D(i * 13.0, py * 0.7) * 0.8;
                        int ix = static_cast<int>(std::lround(px));
                        if (ix < 0 || ix >= static_cast<int>(TILE_SIZE))
                            break;
                        auto& p =
                            getTilePixel(static_cast<uint32_t>(ix), static_cast<uint32_t>(py));
                        p.r = clampToByte(0.55);
                        p.g = clampToByte(0.4);
                        p.b = clampToByte(0.22);
                        p.a = 255;
                    }
                }
                break;
            }

            case BlockType::TALL_GRASS: {
                // One-pixel blades of varying height and green
                for (int blade = 0; blade < 9; ++blade) {
                    uint32_t bx = static_cast<uint32_t>(1 + (blade * 7) % 14);
                    double h = 6.0 + noise.noise2D(blade * 3.1, 0.5) * 4.0;
                    double green = 0.5 + noise.noise2D(blade * 5.7, 9.0) * 0.15;
                    for (uint32_t py = 15; py > 15 - static_cast<uint32_t>(h); --py) {
                        // Lean the blade tip sideways
                        uint32_t px = bx + (py < 10 && blade % 2 == 0 ? 1 : 0);
                        if (px >= TILE_SIZE)
                            continue;
                        auto& p = getTilePixel(px, py);
                        p.r = clampToByte(0.25);
                        p.g = clampToByte(green);
                        p.b = clampToByte(0.15);
                        p.a = 255;
                    }
                }
                break;
            }

            case BlockType::FLOWER_YELLOW:
            case BlockType::FLOWER_RED:
            case BlockType::FLOWER_BLUE: {
                bool red = static_cast<BlockType>(layer) == BlockType::FLOWER_RED;
                bool blue = static_cast<BlockType>(layer) == BlockType::FLOWER_BLUE;
                // Stem: bottom half, center column with a leaf nub
                for (uint32_t py = 8; py < TILE_SIZE; ++py) {
                    auto& p = getTilePixel(7, py);
                    p.r = clampToByte(0.2);
                    p.g = clampToByte(0.5);
                    p.b = clampToByte(0.15);
                    p.a = 255;
                }
                getTilePixel(6, 11) = getTilePixel(7, 11);
                // Petal blob + center dot at the top
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (std::abs(dx) + std::abs(dy) > 3)
                            continue;
                        auto& p = getTilePixel(static_cast<uint32_t>(7 + dx),
                                               static_cast<uint32_t>(5 + dy));
                        p.r = clampToByte(blue ? 0.25 : (red ? 0.85 : 0.9));
                        p.g = clampToByte(blue ? 0.45 : (red ? 0.15 : 0.8));
                        p.b = clampToByte(blue ? 0.90 : (red ? 0.15 : 0.2));
                        p.a = 255;
                    }
                }
                auto& center = getTilePixel(7, 5);
                center.r = clampToByte(red ? 0.95 : 0.6);
                center.g = clampToByte(red ? 0.85 : 0.4);
                center.b = clampToByte(0.2);
                break;
            }

            case BlockType::FERN:
            case BlockType::SHRUB:
            case BlockType::CATTAIL:
            case BlockType::SUCCULENT: {
                double r = 0.18, g = 0.48, b = 0.16;
                if (static_cast<BlockType>(layer) == BlockType::CATTAIL) {
                    r = 0.42;
                    g = 0.48;
                    b = 0.18;
                } else if (static_cast<BlockType>(layer) == BlockType::SUCCULENT) {
                    r = 0.28;
                    g = 0.52;
                    b = 0.34;
                }
                for (uint32_t py = 3; py < TILE_SIZE; ++py) {
                    int halfWidth = std::max(0, 5 - std::abs(11 - static_cast<int>(py)) / 2);
                    for (int dx = -halfWidth; dx <= halfWidth; ++dx) {
                        uint32_t px = static_cast<uint32_t>(std::clamp(7 + dx, 0, 15));
                        fillTilePixel(&getTilePixel(px, py), r, g, b,
                                      noise.noise2D(px * 0.4, py * 0.4), 0.15);
                    }
                }
                break;
            }

            case BlockType::LILY_PAD: {
                for (uint32_t y = 2; y < 14; ++y) {
                    for (uint32_t x = 2; x < 14; ++x) {
                        double dx = static_cast<double>(x) - 7.5;
                        double dy = static_cast<double>(y) - 7.5;
                        if (dx * dx + dy * dy > 34.0 || (x >= 8 && y <= 7))
                            continue;
                        fillTilePixel(&getTilePixel(x, y), 0.18, 0.48, 0.16,
                                      noise.noise2D(x * 0.3, y * 0.3), 0.12);
                    }
                }
                break;
            }

            case BlockType::MUSHROOM_BROWN:
            case BlockType::MUSHROOM_RED: {
                bool red = static_cast<BlockType>(layer) == BlockType::MUSHROOM_RED;
                // Stem: two-pixel pale column, lower half
                for (uint32_t py = 9; py < TILE_SIZE; ++py) {
                    for (uint32_t px = 7; px <= 8; ++px) {
                        auto& p = getTilePixel(px, py);
                        p.r = clampToByte(0.85);
                        p.g = clampToByte(0.8);
                        p.b = clampToByte(0.7);
                        p.a = 255;
                    }
                }
                // Dome cap
                for (int dy = 0; dy <= 3; ++dy) {
                    int halfWidth = 4 - dy;
                    for (int dx = -halfWidth; dx <= halfWidth; ++dx) {
                        auto& p = getTilePixel(static_cast<uint32_t>(7 + dx),
                                               static_cast<uint32_t>(9 - dy));
                        p.r = clampToByte(red ? 0.8 : 0.55);
                        p.g = clampToByte(red ? 0.15 : 0.4);
                        p.b = clampToByte(red ? 0.15 : 0.3);
                        p.a = 255;
                    }
                }
                if (red) {
                    // White spots on the red cap
                    getTilePixel(5, 8).r = getTilePixel(5, 8).g = getTilePixel(5, 8).b = 240;
                    getTilePixel(9, 7).r = getTilePixel(9, 7).g = getTilePixel(9, 7).b = 240;
                }
                break;
            }

            case BlockType::REED: {
                // Full-height stalks with darker node rows
                for (int stalk = 0; stalk < 3; ++stalk) {
                    uint32_t sx = static_cast<uint32_t>(3 + stalk * 5);
                    for (uint32_t py = 0; py < TILE_SIZE; ++py) {
                        bool node = (py % 6 == 5);
                        double f = node ? 0.7 : 1.0;
                        for (uint32_t px = sx; px <= sx + 1; ++px) {
                            auto& p = getTilePixel(px, py);
                            p.r = clampToByte(0.45 * f);
                            p.g = clampToByte(0.65 * f);
                            p.b = clampToByte(0.3 * f);
                            p.a = 255;
                        }
                    }
                }
                break;
            }

            case BlockType::LAVA: {
                // Orange base with bright veins; vein luminance ≥ 1.0 so the
                // bloom pass makes lava glow
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double vein = noise.noise2D(x * 0.35, y * 0.35);
                        if (vein > 0.4) {
                            auto& p = getTilePixel(x, y);
                            p.r = 255;
                            p.g = clampToByte(0.9);
                            p.b = clampToByte(0.3);
                            p.a = 255;
                        } else {
                            double n = noise.noise2D(x * 0.2 + 10.0, y * 0.2 + 10.0);
                            fillTilePixel(&getTilePixel(x, y), 0.9, 0.35, 0.05, n, 0.15);
                        }
                    }
                }
                break;
            }

            case BlockType::ICE: {
                // Pale blue with noise-threshold streaks (opaque v1)
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.2, y * 0.2);
                        double streak = noise.noise2D(x * 0.5 + 70.0, y * 0.15 + 70.0);
                        double f = streak > 0.5 ? 1.1 : 1.0;
                        fillTilePixel(&getTilePixel(x, y), 0.72 * f, 0.84 * f, 0.95 * f, n, 0.04);
                    }
                }
                break;
            }

            case BlockType::CRAFTING_TABLE: {
                // Plank side with a dark tool band across the top rows
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.15, y * 0.15);
                        bool band = y < 4;
                        bool slot = band && (x % 5 == 2);
                        double f = slot ? 0.35 : (band ? 0.7 : 1.0);
                        fillTilePixel(&getTilePixel(x, y), 0.6 * f, 0.45 * f, 0.28 * f, n, 0.06);
                    }
                }
                break;
            }

            case BlockType::FURNACE:
            case BlockType::FURNACE_LIT: {
                // Cobble-style speckle with the mouth cut into the lower half
                for (uint32_t y = 0; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 0; x < TILE_SIZE; ++x) {
                        double n = noise.noise2D(x * 0.3, y * 0.3);
                        fillTilePixel(&getTilePixel(x, y), 0.42, 0.42, 0.44, n, 0.14);
                    }
                }
                const bool lit = static_cast<BlockType>(layer) == BlockType::FURNACE_LIT;
                for (uint32_t y = 6; y < 13; ++y) {
                    for (uint32_t x = 3; x < 13; ++x) {
                        auto& p = getTilePixel(x, y);
                        if (lit) {
                            double flame = noise.noise2D(x * 0.6 + 30.0, y * 0.6 + 30.0);
                            p.r = 255;
                            p.g = clampToByte(0.55 + flame * 0.25);
                            p.b = clampToByte(0.08);
                        } else {
                            p.r = p.g = p.b = clampToByte(0.12);
                        }
                        p.a = 255;
                    }
                }
                break;
            }

            case BlockType::TORCH: {
                // Cutout cross: a glowing tip (row 0 is the tile top) over a
                // thin stick column; everything else stays transparent
                for (uint32_t y = 5; y < TILE_SIZE; ++y) {
                    for (uint32_t x = 7; x < 9; ++x) {
                        double n = noise.noise2D(x * 0.4, y * 0.4);
                        fillTilePixel(&getTilePixel(x, y), 0.45, 0.3, 0.15, n, 0.1);
                    }
                }
                for (uint32_t y = 1; y < 5; ++y) {
                    for (uint32_t x = 6; x < 10; ++x) {
                        auto& p = getTilePixel(x, y);
                        double flame = noise.noise2D(x * 0.7, y * 0.7);
                        p.r = 255;
                        p.g = clampToByte(0.85 + flame * 0.1);
                        p.b = clampToByte(0.35);
                        p.a = 255;
                    }
                }
                break;
            }

            // AIR (never drawn) gets a transparent black layer
            default:
                break;
        }
    }

    uploadLayerMips(_texture, tilePixels, layer);
}

static void uploadLayerMips(id<MTLTexture> texture,
                            const std::array<BgraPixel, BlockTextureArray::TILE_SIZE *
                                                            BlockTextureArray::TILE_SIZE>& pixels,
                            uint8_t layer) {
    std::vector<BgraPixel> mipPixels(pixels.begin(), pixels.end());
    const uint32_t baseCovered = alphaCoverage(mipPixels);
    constexpr uint32_t BASE_TEXEL_COUNT =
        BlockTextureArray::TILE_SIZE * BlockTextureArray::TILE_SIZE;
    uint32_t mipEdge = BlockTextureArray::TILE_SIZE;

    for (uint32_t mipLevel = 0; mipLevel < BlockTextureArray::MIP_LEVEL_COUNT; ++mipLevel) {
        [texture replaceRegion:MTLRegionMake2D(0, 0, mipEdge, mipEdge)
                   mipmapLevel:mipLevel
                         slice:layer
                     withBytes:mipPixels.data()
                   bytesPerRow:mipEdge * sizeof(BgraPixel)
                 bytesPerImage:0];

        if (mipLevel + 1 == BlockTextureArray::MIP_LEVEL_COUNT)
            break;
        mipPixels = downsampleAlphaAware(mipPixels, mipEdge);
        preserveAlphaCoverage(mipPixels, baseCovered, BASE_TEXEL_COUNT);
        mipEdge /= 2;
    }
}

// ---------------------------------------------------------------------------
// Item icons - procedural 16x16 pixel art per non-block item: geometric
// painters parameterized by the item's palette, so twelve tools come from
// four shapes and the meats from three.
// ---------------------------------------------------------------------------
static void paintItemIcon(BgraPixel* pixels, ItemType item, SimplexNoise& noise) {
    constexpr int TILE = static_cast<int>(BlockTextureArray::TILE_SIZE);
    auto setPx = [&](int x, int y, uint32_t rgb, double shadeJitter = 0.08) {
        if (x < 0 || x >= TILE || y < 0 || y >= TILE)
            return;
        const double n = noise.noise2D(x * 0.5, y * 0.5) * shadeJitter;
        BgraPixel& p = pixels[static_cast<uint32_t>(y) * BlockTextureArray::TILE_SIZE +
                              static_cast<uint32_t>(x)];
        p.r = clampToByte(((rgb >> 16) & 0xFF) / 255.0 * (1.0 + n));
        p.g = clampToByte(((rgb >> 8) & 0xFF) / 255.0 * (1.0 + n));
        p.b = clampToByte((rgb & 0xFF) / 255.0 * (1.0 + n));
        p.a = 255;
    };
    auto fillRect = [&](int x0, int y0, int x1, int y1, uint32_t rgb) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                setPx(x, y, rgb);
            }
        }
    };
    // A 2px-wide diagonal from (x0, y0) up-right to (x1, y1).
    auto fillDiagonal = [&](int x0, int y0, int x1, int y1, uint32_t rgb) {
        const int steps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
        for (int i = 0; i <= steps; ++i) {
            const int x = x0 + (x1 - x0) * i / std::max(1, steps);
            const int y = y0 + (y1 - y0) * i / std::max(1, steps);
            setPx(x, y, rgb);
            setPx(x + 1, y, rgb);
        }
    };
    auto fillEllipse = [&](int cx, int cy, int rx, int ry, uint32_t rgb) {
        for (int y = cy - ry; y <= cy + ry; ++y) {
            for (int x = cx - rx; x <= cx + rx; ++x) {
                const double dx = static_cast<double>(x - cx) / rx;
                const double dy = static_cast<double>(y - cy) / ry;
                if (dx * dx + dy * dy <= 1.0)
                    setPx(x, y, rgb);
            }
        }
    };

    constexpr uint32_t WOOD = 0x8A5C2E;
    const uint32_t swatch = itemSwatchColor(item);
    const ItemDefinition definition = itemDefinition(item);

    if (definition.category == ItemCategory::TOOL) {
        // Handle first, head painted over it in the tier material.
        fillDiagonal(3, 12, 11, 4, WOOD);
        switch (definition.toolClass) {
            case ToolClass::PICKAXE:
                fillRect(3, 2, 12, 3, swatch);
                fillRect(2, 3, 4, 5, swatch);
                fillRect(11, 3, 13, 5, swatch);
                break;
            case ToolClass::AXE:
                fillRect(8, 1, 12, 5, swatch);
                fillRect(6, 2, 8, 4, swatch);
                break;
            case ToolClass::SHOVEL:
                fillEllipse(11, 3, 3, 3, swatch);
                break;
            case ToolClass::SWORD:
                fillDiagonal(5, 10, 12, 3, swatch);
                fillDiagonal(6, 11, 13, 4, swatch);
                setPx(4, 11, 0x33261A);
                setPx(5, 12, 0x33261A);
                break;
            case ToolClass::NONE:
                break;
        }
        return;
    }

    if (definition.category == ItemCategory::FOOD) {
        const bool fish = item == ItemType::RAW_FISH || item == ItemType::COOKED_FISH;
        const bool bird = item == ItemType::RAW_CHICKEN || item == ItemType::COOKED_CHICKEN;
        if (fish) {
            fillEllipse(7, 8, 5, 3, swatch);
            for (int i = 0; i < 4; ++i) {
                fillRect(11 + i, 8 - i, 11 + i, 8 + i, swatch); // tail fan
            }
            setPx(4, 7, 0x1A1A1A); // eye
        } else if (bird) {
            fillEllipse(9, 6, 4, 4, swatch);
            fillDiagonal(3, 13, 7, 9, 0xF2EFE6); // bone
            fillEllipse(3, 13, 1, 1, 0xF2EFE6);
        } else {
            fillEllipse(8, 8, 5, 4, swatch);
            fillRect(5, 7, 10, 8, 0xF2E6D9); // marbling
        }
        return;
    }

    switch (item) {
        case ItemType::STICK:
            fillDiagonal(4, 12, 11, 4, swatch);
            break;
        case ItemType::COAL:
        case ItemType::CHARCOAL:
            fillEllipse(8, 8, 4, 4, swatch);
            setPx(6, 7, 0x000000);
            setPx(9, 9, 0x000000);
            break;
        case ItemType::IRON_INGOT:
        case ItemType::GOLD_INGOT: {
            fillRect(3, 8, 12, 12, swatch); // front face
            for (int i = 0; i < 3; ++i) {
                fillRect(4 + i, 7 - i, 13 + i - 3, 7 - i, swatch); // beveled top
            }
            fillRect(4, 6, 11, 6, (swatch & 0xFFFFFF) | 0x303030); // top sheen
            break;
        }
        case ItemType::DIAMOND: {
            for (int y = 4; y <= 12; ++y) {
                const int half = y <= 7 ? (y - 3) * 2 : (13 - y);
                fillRect(8 - half, y, 8 + half - 1, y, swatch);
            }
            break;
        }
        case ItemType::BUCKET:
        case ItemType::WATER_BUCKET:
        case ItemType::LAVA_BUCKET: {
            constexpr uint32_t METAL = 0xB0B4BA;
            // A filled bucket carries its fluid in the mouth of the pail.
            if (item != ItemType::BUCKET) {
                const uint32_t fluid = item == ItemType::WATER_BUCKET ? 0x4073D9 : 0xE6661A;
                fillRect(5, 5, 10, 7, fluid);
            }
            fillRect(4, 4, 11, 5, METAL); // rim
            for (int y = 6; y <= 12; ++y) {
                const int inset = (y - 6) / 3; // taper toward a narrow base
                fillRect(4 + inset, y, 11 - inset, y, METAL);
            }
            break;
        }
        default:
            fillEllipse(8, 8, 4, 4, swatch);
            break;
    }
}

BlockTextureArray::BlockTextureArray(id<MTLDevice> device) {
    auto descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = MTLTextureType2DArray;
    descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    descriptor.width = TILE_SIZE;
    descriptor.height = TILE_SIZE;
    descriptor.arrayLength = TEXTURE_LAYER_TOTAL;
    descriptor.mipmapLevelCount = MIP_LEVEL_COUNT;
    descriptor.usage = MTLTextureUsageShaderRead;

    _texture = [device newTextureWithDescriptor:descriptor];
    if (!_texture) {
        RY_LOG_FATAL("Failed to allocate block texture array");
    }

    // Repeat addressing is what lets a single greedy quad tile its texture
    // across every block it covers.
    auto samplerDesc = [[MTLSamplerDescriptor alloc] init];
    samplerDesc.magFilter = MTLSamplerMinMagFilterNearest;
    samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDesc.mipFilter = MTLSamplerMipFilterLinear;
    samplerDesc.maxAnisotropy = MAX_ANISOTROPY;
    samplerDesc.sAddressMode = MTLSamplerAddressModeRepeat;
    samplerDesc.tAddressMode = MTLSamplerAddressModeRepeat;
    _sampler = [device newSamplerStateWithDescriptor:samplerDesc];
    if (!_sampler) {
        RY_LOG_FATAL("Failed to create block texture sampler");
    }

    for (uint16_t layer = 0; layer < TEXTURE_LAYER_TOTAL; ++layer) {
        generateLayer(static_cast<uint8_t>(layer));
    }
}

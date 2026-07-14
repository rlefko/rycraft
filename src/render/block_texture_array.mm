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
            case BlockType::SPRUCE_LEAVES: {
                double leafR = 0.2, leafG = 0.5, leafB = 0.15; // oak
                if (static_cast<BlockType>(layer) == BlockType::BIRCH_LEAVES) {
                    leafR = 0.35;
                    leafG = 0.55;
                    leafB = 0.25;
                } else if (static_cast<BlockType>(layer) == BlockType::SPRUCE_LEAVES) {
                    leafR = 0.12;
                    leafG = 0.35;
                    leafB = 0.18;
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
            case BlockType::FLOWER_RED: {
                bool red = static_cast<BlockType>(layer) == BlockType::FLOWER_RED;
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
                        p.r = clampToByte(red ? 0.85 : 0.9);
                        p.g = clampToByte(red ? 0.15 : 0.8);
                        p.b = clampToByte(red ? 0.15 : 0.2);
                        p.a = 255;
                    }
                }
                auto& center = getTilePixel(7, 5);
                center.r = clampToByte(red ? 0.95 : 0.6);
                center.g = clampToByte(red ? 0.85 : 0.4);
                center.b = clampToByte(0.2);
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

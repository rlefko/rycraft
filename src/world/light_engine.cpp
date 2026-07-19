#include "world/light_engine.hpp"

#include "world/block_properties.hpp"

#include <algorithm>
#include <vector>

LightEngine::FloodResult LightEngine::floodChunk(Chunk& chunk, const FaceNeighbors& neighbors,
                                                 const SkyLightSeedColumns& skySeeds) {
    // A frontier item stores cellIndex * 2 + channel, where zero is block
    // light and one is skylight. A cell is queued only when that channel
    // rises, so both sweeps end at their unique max-light fixed points.
    std::vector<uint8_t> light;
    std::vector<int> frontier;

    auto raise = [&](int x, int y, int z, uint8_t level, bool sky) {
        if (light.empty()) {
            light.assign(CHUNK_VOLUME, 0);
        }
        const int i = Chunk::index(x, y, z);
        const uint8_t current = sky ? derivedSkyLight(light[i]) : derivedBlockLight(light[i]);
        if (level > current) {
            light[i] = sky ? packDerivedLight(level, derivedBlockLight(light[i]))
                           : packDerivedLight(derivedSkyLight(light[i]), level);
            frontier.push_back(i * 2 + static_cast<int>(sky));
        }
    };

    const int32_t cubeBaseY = chunk.chunkY * CHUNK_EDGE;
    for (int z = 0; z < CHUNK_DEPTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            const int32_t cutoffY = skySeeds.at(x, z);
            const int firstLocalY = std::max(0, cutoffY - cubeBaseY);
            for (int y = firstLocalY; y < CHUNK_HEIGHT; ++y) {
                if (isTransparent(chunk.getBlock(x, y, z))) {
                    raise(x, y, z, 15, true);
                }
            }
        }
    }

    // Seed self emitters (lava). The emitter cell itself may be opaque; light
    // still radiates from it into the transparent cells around it.
    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < CHUNK_DEPTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                uint8_t emission = blockLightEmission(chunk.getBlock(x, y, z));
                if (emission > 0) {
                    raise(x, y, z, emission, false);
                }
            }
        }
    }

    // Seed the six border planes from the neighbors' adjacent cells
    // (their light minus one), only into transparent cells that can receive it.
    auto seedX = [&](const Chunk* n, int borderX, int neighborX) {
        if (!n) {
            return;
        }
        for (int y = 0; y < CHUNK_HEIGHT; ++y) {
            for (int z = 0; z < CHUNK_DEPTH; ++z) {
                if (!isTransparent(chunk.getBlock(borderX, y, z))) {
                    continue;
                }
                const uint8_t incomingBlock = n->getBlockLight(neighborX, y, z);
                const uint8_t incomingSky = n->getSkyLight(neighborX, y, z);
                if (incomingBlock > 1) {
                    raise(borderX, y, z, static_cast<uint8_t>(incomingBlock - 1), false);
                }
                if (incomingSky > 1) {
                    raise(borderX, y, z, static_cast<uint8_t>(incomingSky - 1), true);
                }
            }
        }
    };
    auto seedZ = [&](const Chunk* n, int borderZ, int neighborZ) {
        if (!n) {
            return;
        }
        for (int y = 0; y < CHUNK_HEIGHT; ++y) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                if (!isTransparent(chunk.getBlock(x, y, borderZ))) {
                    continue;
                }
                const uint8_t incomingBlock = n->getBlockLight(x, y, neighborZ);
                const uint8_t incomingSky = n->getSkyLight(x, y, neighborZ);
                if (incomingBlock > 1) {
                    raise(x, y, borderZ, static_cast<uint8_t>(incomingBlock - 1), false);
                }
                if (incomingSky > 1) {
                    raise(x, y, borderZ, static_cast<uint8_t>(incomingSky - 1), true);
                }
            }
        }
    };
    auto seedY = [&](const Chunk* n, int borderY, int neighborY) {
        if (!n) {
            return;
        }
        for (int z = 0; z < CHUNK_DEPTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                if (!isTransparent(chunk.getBlock(x, borderY, z))) {
                    continue;
                }
                const uint8_t incomingBlock = n->getBlockLight(x, neighborY, z);
                const uint8_t incomingSky = n->getSkyLight(x, neighborY, z);
                if (incomingBlock > 1) {
                    raise(x, borderY, z, static_cast<uint8_t>(incomingBlock - 1), false);
                }
                if (incomingSky > 1) {
                    raise(x, borderY, z, static_cast<uint8_t>(incomingSky - 1), true);
                }
            }
        }
    };
    seedX(neighbors[0], 0, CHUNK_WIDTH - 1);  // -X neighbor's +X wall
    seedX(neighbors[1], CHUNK_WIDTH - 1, 0);  // +X neighbor's -X wall
    seedZ(neighbors[2], 0, CHUNK_DEPTH - 1);  // -Z neighbor's +Z wall
    seedZ(neighbors[3], CHUNK_DEPTH - 1, 0);  // +Z neighbor's -Z wall
    seedY(neighbors[4], 0, CHUNK_HEIGHT - 1); // -Y neighbor's +Y wall
    seedY(neighbors[5], CHUNK_HEIGHT - 1, 0); // +Y neighbor's -Y wall

    // Flood inward: each lit cell spills to its six in-chunk transparent
    // neighbors at one level lower.
    static constexpr int DX[6] = {1, -1, 0, 0, 0, 0};
    static constexpr int DY[6] = {0, 0, 1, -1, 0, 0};
    static constexpr int DZ[6] = {0, 0, 0, 0, 1, -1};
    for (size_t head = 0; head < frontier.size(); ++head) {
        const int item = frontier[head];
        const int i = item / 2;
        const bool sky = (item & 1) != 0;
        const uint8_t level = sky ? derivedSkyLight(light[i]) : derivedBlockLight(light[i]);
        if (level <= 1) {
            continue;
        }
        int x = i % CHUNK_WIDTH;
        int z = (i / CHUNK_WIDTH) % CHUNK_DEPTH;
        int y = i / (CHUNK_WIDTH * CHUNK_DEPTH);
        for (int d = 0; d < 6; ++d) {
            int nx = x + DX[d], ny = y + DY[d], nz = z + DZ[d];
            if (nx < 0 || nx >= CHUNK_WIDTH || ny < 0 || ny >= CHUNK_HEIGHT || nz < 0 ||
                nz >= CHUNK_DEPTH) {
                continue;
            }
            if (!isTransparent(chunk.getBlock(nx, ny, nz))) {
                continue;
            }
            raise(nx, ny, nz, static_cast<uint8_t>(level - 1), sky);
        }
    }

    const std::vector<uint8_t>& previous = chunk.packedLightData();
    if (previous == light) {
        return {};
    }

    const auto levelAt = [](const std::vector<uint8_t>& field, int x, int y, int z) {
        return field.empty() ? uint8_t{0} : field[Chunk::index(x, y, z)];
    };
    uint8_t changedFaces = 0;
    const auto compareX = [&](int x) {
        for (int z = 0; z < CHUNK_DEPTH; ++z) {
            for (int y = 0; y < CHUNK_HEIGHT; ++y) {
                if (levelAt(previous, x, y, z) != levelAt(light, x, y, z)) {
                    return true;
                }
            }
        }
        return false;
    };
    const auto compareZ = [&](int z) {
        for (int y = 0; y < CHUNK_HEIGHT; ++y) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                if (levelAt(previous, x, y, z) != levelAt(light, x, y, z)) {
                    return true;
                }
            }
        }
        return false;
    };
    const auto compareY = [&](int y) {
        for (int z = 0; z < CHUNK_DEPTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                if (levelAt(previous, x, y, z) != levelAt(light, x, y, z)) {
                    return true;
                }
            }
        }
        return false;
    };
    if (compareX(0)) {
        changedFaces |= CHANGED_MINUS_X;
    }
    if (compareX(CHUNK_WIDTH - 1)) {
        changedFaces |= CHANGED_PLUS_X;
    }
    if (compareZ(0)) {
        changedFaces |= CHANGED_MINUS_Z;
    }
    if (compareZ(CHUNK_DEPTH - 1)) {
        changedFaces |= CHANGED_PLUS_Z;
    }
    if (compareY(0)) {
        changedFaces |= CHANGED_MINUS_Y;
    }
    if (compareY(CHUNK_HEIGHT - 1)) {
        changedFaces |= CHANGED_PLUS_Y;
    }

    // Commit, keeping the array unallocated when the chunk is fully dark.
    if (light.empty()) {
        chunk.clearDerivedLight();
        return {.changedState = true, .changedFaceMask = changedFaces};
    }
    chunk.replacePackedLight(std::move(light));
    return {.changedState = true, .changedFaceMask = changedFaces};
}

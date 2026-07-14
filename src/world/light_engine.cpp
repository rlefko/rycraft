#include "world/light_engine.hpp"

#include "world/block_properties.hpp"

#include <vector>

namespace {

// Same Y-major layout as Chunk::blocks (x + z*W + y*W*D).
inline int lightIndex(int x, int y, int z) {
    return x + z * CHUNK_WIDTH + y * CHUNK_WIDTH * CHUNK_DEPTH;
}

} // namespace

bool LightEngine::floodChunk(Chunk& chunk, const FaceNeighbors& neighbors) {
    // Built lazily: a chunk with no light stays fully dark and allocates
    // nothing. `frontier` is a FIFO of lit cell indices to expand; a cell is
    // re-queued only when its level rises, so the sweep ends at the max light.
    std::vector<uint8_t> light;
    std::vector<int> frontier;

    auto raise = [&](int x, int y, int z, uint8_t level) {
        if (light.empty()) {
            light.assign(CHUNK_VOLUME, 0);
        }
        int i = lightIndex(x, y, z);
        if (level > light[i]) {
            light[i] = level;
            frontier.push_back(i);
        }
    };

    // Seed self emitters (lava). The emitter cell itself may be opaque; light
    // still radiates from it into the transparent cells around it.
    for (int y = 0; y < CHUNK_HEIGHT; ++y) {
        for (int z = 0; z < CHUNK_DEPTH; ++z) {
            for (int x = 0; x < CHUNK_WIDTH; ++x) {
                uint8_t emission = blockLightEmission(chunk.getBlock(x, y, z));
                if (emission > 0) {
                    raise(x, y, z, emission);
                }
            }
        }
    }

    // Seed the four border planes from the neighbors' adjacent columns
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
                uint8_t incoming = n->getBlockLight(neighborX, y, z);
                if (incoming > 1) {
                    raise(borderX, y, z, static_cast<uint8_t>(incoming - 1));
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
                uint8_t incoming = n->getBlockLight(x, y, neighborZ);
                if (incoming > 1) {
                    raise(x, y, borderZ, static_cast<uint8_t>(incoming - 1));
                }
            }
        }
    };
    seedX(neighbors[0], 0, CHUNK_WIDTH - 1); // -X neighbor's +X wall
    seedX(neighbors[1], CHUNK_WIDTH - 1, 0); // +X neighbor's -X wall
    seedZ(neighbors[2], 0, CHUNK_DEPTH - 1); // -Z neighbor's +Z wall
    seedZ(neighbors[3], CHUNK_DEPTH - 1, 0); // +Z neighbor's -Z wall

    // Flood inward: each lit cell spills to its six in-chunk transparent
    // neighbors at one level lower.
    static constexpr int DX[6] = {1, -1, 0, 0, 0, 0};
    static constexpr int DY[6] = {0, 0, 1, -1, 0, 0};
    static constexpr int DZ[6] = {0, 0, 0, 0, 1, -1};
    for (size_t head = 0; head < frontier.size(); ++head) {
        int i = frontier[head];
        int level = light[i];
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
            raise(nx, ny, nz, static_cast<uint8_t>(level - 1));
        }
    }

    // Commit, keeping the array unallocated when the chunk is fully dark.
    if (light.empty()) {
        bool changed = !chunk.blockLight.empty();
        chunk.blockLight.clear();
        return changed;
    }
    if (chunk.blockLight == light) {
        return false;
    }
    chunk.blockLight = std::move(light);
    return true;
}

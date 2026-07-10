#pragma once

constexpr int CHUNK_SIZE = 16;

enum class BlockType : unsigned char {
  AIR = 0,
  DIRT = 1,
  GRASS = 2,
  STONE = 3,
};

#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <world/chunk.hpp>

TEST_CASE("Vec3 zero returns all-zero vector", "[math]") {
  Vec3 v = Vec3::zero();
  REQUIRE(v.x == 0.f);
  REQUIRE(v.y == 0.f);
  REQUIRE(v.z == 0.f);
}

TEST_CASE("Chunk coordinates are multiples of CHUNK_SIZE", "[chunk]") {
  REQUIRE(0 % CHUNK_SIZE == 0);
  REQUIRE(16 % CHUNK_SIZE == 0);
  REQUIRE(32 % CHUNK_SIZE == 0);
  REQUIRE((-16) % CHUNK_SIZE == 0);
}

TEST_CASE("BlockType enum values are as expected", "[block]") {
  REQUIRE(static_cast<int>(BlockType::AIR) == 0);
  REQUIRE(static_cast<int>(BlockType::DIRT) == 1);
  REQUIRE(static_cast<int>(BlockType::GRASS) == 2);
  REQUIRE(static_cast<int>(BlockType::STONE) == 3);
}

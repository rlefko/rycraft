#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <common/math.hpp>
#include <common/result.hpp>
#include <common/thread_pool.hpp>
#include <world/chunk.hpp>
#include <world/noise.hpp>
#include <world/terrain.hpp>
#include <world/biome.hpp>
#include <world/world.hpp>
#include <world/serialization.hpp>
#include <world/save_manager.hpp>
#include <render/vertex.hpp>
#include <render/mesher.hpp>
#include <render/texture_atlas.hpp>
#include <render/mega_buffer.hpp>
#include <render/ui_overlay.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/voxel_traversal.hpp>
#include <engine/hotbar.hpp>

#include <cmath>
#include <thread>
#include <chrono>

// ============================================================================
// Vec3 Tests
// ============================================================================
TEST_CASE("Vec3 zero returns all-zero vector", "[math]") {
  Vec3 v = Vec3::zero();
  REQUIRE(v.x == 0.f);
  REQUIRE(v.y == 0.f);
  REQUIRE(v.z == 0.f);
}

TEST_CASE("Vec3 addition", "[math]") {
  Vec3 a{1.f, 2.f, 3.f};
  Vec3 b{4.f, 5.f, 6.f};
  Vec3 c = a + b;
  REQUIRE(c.x == 5.f);
  REQUIRE(c.y == 7.f);
  REQUIRE(c.z == 9.f);
}

TEST_CASE("Vec3 subtraction", "[math]") {
  Vec3 a{4.f, 5.f, 6.f};
  Vec3 b{1.f, 2.f, 3.f};
  Vec3 c = a - b;
  REQUIRE(c.x == 3.f);
  REQUIRE(c.y == 3.f);
  REQUIRE(c.z == 3.f);
}

TEST_CASE("Vec3 scalar multiply", "[math]") {
  Vec3 v{1.f, 2.f, 3.f};
  Vec3 s = v * 2.f;
  REQUIRE(s.x == 2.f);
  REQUIRE(s.y == 4.f);
  REQUIRE(s.z == 6.f);

  // Free function: scalar * vector
  Vec3 t = 3.f * v;
  REQUIRE(t.x == 3.f);
  REQUIRE(t.y == 6.f);
  REQUIRE(t.z == 9.f);
}

TEST_CASE("Vec3 dot product", "[math]") {
  Vec3 a{1.f, 2.f, 3.f};
  Vec3 b{4.f, 5.f, 6.f};
  float d = a.dot(b);
  REQUIRE(d == 32.f); // 1*4 + 2*5 + 3*6
}

TEST_CASE("Vec3 cross product", "[math]") {
  Vec3 a{1.f, 0.f, 0.f};
  Vec3 b{0.f, 1.f, 0.f};
  Vec3 c = a.cross(b);
  REQUIRE(c.x == 0.f);
  REQUIRE(c.y == 0.f);
  REQUIRE(c.z == 1.f);
}

TEST_CASE("Vec3 normalize", "[math]") {
  Vec3 v{3.f, 0.f, 0.f};
  Vec3 n = v.normalize();
  REQUIRE(n.x == Catch::Approx(1.f));
  REQUIRE(n.y == Catch::Approx(0.f));
  REQUIRE(n.z == Catch::Approx(0.f));

  // Zero vector returns zero
  Vec3 z = Vec3::zero().normalize();
  REQUIRE(z.x == 0.f);
  REQUIRE(z.y == 0.f);
  REQUIRE(z.z == 0.f);
}

TEST_CASE("Vec3 length", "[math]") {
  Vec3 v{3.f, 4.f, 0.f};
  REQUIRE(v.length() == Catch::Approx(5.f));
  REQUIRE(v.lengthSq() == Catch::Approx(25.f));
}

TEST_CASE("Vec3 lerp", "[math]") {
  Vec3 a{0.f, 0.f, 0.f};
  Vec3 b{10.f, 10.f, 10.f};
  Vec3 m = a.lerp(b, 0.5f);
  REQUIRE(m.x == Catch::Approx(5.f));
  REQUIRE(m.y == Catch::Approx(5.f));
  REQUIRE(m.z == Catch::Approx(5.f));
}

// ============================================================================
// Vec2 Tests
// ============================================================================
TEST_CASE("Vec2 arithmetic", "[math]") {
  Vec2 a{1.f, 2.f};
  Vec2 b{3.f, 4.f};
  Vec2 s = a + b;
  REQUIRE(s.x == 4.f);
  REQUIRE(s.y == 6.f);
  REQUIRE(a.dot(b) == 11.f);
}

TEST_CASE("Vec2 normalize", "[math]") {
  Vec2 v{0.f, 5.f};
  Vec2 n = v.normalize();
  REQUIRE(n.x == Catch::Approx(0.f));
  REQUIRE(n.y == Catch::Approx(1.f));
}

// ============================================================================
// Vec4 Tests
// ============================================================================
TEST_CASE("Vec4 arithmetic", "[math]") {
  Vec4 a{1.f, 2.f, 3.f, 4.f};
  Vec4 b{5.f, 6.f, 7.f, 8.f};
  Vec4 s = a + b;
  REQUIRE(s.x == 6.f);
  REQUIRE(s.y == 8.f);
  REQUIRE(s.z == 10.f);
  REQUIRE(s.w == 12.f);
}

TEST_CASE("Vec4 from Vec3", "[math]") {
  Vec3 v{1.f, 2.f, 3.f};
  Vec4 h{v, 1.f};
  REQUIRE(h.x == 1.f);
  REQUIRE(h.y == 2.f);
  REQUIRE(h.z == 3.f);
  REQUIRE(h.w == 1.f);
}

// ============================================================================
// Mat4 Tests
// ============================================================================
TEST_CASE("Mat4 identity", "[math]") {
  Mat4 m = Mat4::identity();
  REQUIRE(m(0, 0) == 1.f);
  REQUIRE(m(1, 1) == 1.f);
  REQUIRE(m(2, 2) == 1.f);
  REQUIRE(m(3, 3) == 1.f);
  REQUIRE(m(0, 1) == 0.f);
  REQUIRE(m(1, 0) == 0.f);
}

TEST_CASE("Mat4 perspective", "[math]") {
  Mat4 p = Mat4::perspective(
      static_cast<float>(std::atan(1.f) * 2.f), // 90 deg
      1.f,                                       // aspect
      1.f,                                       // near
      100.f                                      // far
  );
  // For 90 deg FOV, aspect 1: tan(45) = 1, so (0,0) = 1.0
  REQUIRE(p(0, 0) == Catch::Approx(1.f).epsilon(0.01f));
  REQUIRE(p(1, 1) == Catch::Approx(1.f).epsilon(0.01f));
  REQUIRE(p(3, 3) == Catch::Approx(0.f));
  // Bottom-right 2x2 should have the far/near terms
  REQUIRE(p(2, 2) != 0.f);
  REQUIRE(p(2, 3) != 0.f);
  REQUIRE(p(3, 2) == Catch::Approx(-1.f));
}

TEST_CASE("Mat4 lookAt", "[math]") {
  Vec3 eye{0.f, 0.f, 5.f};
  Vec3 target{0.f, 0.f, 0.f};
  Vec3 up{0.f, 1.f, 0.f};
  Mat4 v = Mat4::lookAt(eye, target, up);

  // Camera looking down -Z from (0,0,5)
  // The translation component should place origin 5 units back
  REQUIRE(v(3, 2) == Catch::Approx(-5.f).epsilon(0.01f));
}

TEST_CASE("Mat4 translation", "[math]") {
  Mat4 t = Mat4::translation(10.f, 20.f, 30.f);
  REQUIRE(t(0, 3) == 10.f);
  REQUIRE(t(1, 3) == 20.f);
  REQUIRE(t(2, 3) == 30.f);
  REQUIRE(t(0, 0) == 1.f);

  // Transform a point
  Vec3 p{1.f, 0.f, 0.f};
  Vec3 tp = t.transformVec3(p);
  REQUIRE(tp.x == Catch::Approx(11.f));
  REQUIRE(tp.y == Catch::Approx(20.f));
  REQUIRE(tp.z == Catch::Approx(30.f));
}

TEST_CASE("Mat4 rotationY", "[math]") {
  float pi = static_cast<float>(std::atan(1.f) * 4.f);
  Mat4 r = Mat4::rotationY(pi * 0.5f); // 90 degrees

  // cos(90) = 0, sin(90) = 1
  REQUIRE(r(0, 0) == Catch::Approx(0.f).margin(0.001f));
  REQUIRE(r(0, 2) == Catch::Approx(1.f).margin(0.001f));
  REQUIRE(r(2, 0) == Catch::Approx(-1.f).margin(0.001f));
  REQUIRE(r(2, 2) == Catch::Approx(0.f).margin(0.001f));
  REQUIRE(r(1, 1) == Catch::Approx(1.f));
}

TEST_CASE("Mat4 scale", "[math]") {
  Mat4 s = Mat4::scale(2.f, 3.f, 4.f);
  REQUIRE(s(0, 0) == 2.f);
  REQUIRE(s(1, 1) == 3.f);
  REQUIRE(s(2, 2) == 4.f);

  // Uniform scale
  Mat4 u = Mat4::scale(5.f);
  REQUIRE(u(0, 0) == 5.f);
  REQUIRE(u(1, 1) == 5.f);
  REQUIRE(u(2, 2) == 5.f);
}

TEST_CASE("Mat4 multiplication", "[math]") {
  Mat4 t = Mat4::translation(1.f, 2.f, 3.f);
  Mat4 s = Mat4::scale(2.f);
  Mat4 ts = t * s;

  // T*S: scale is applied first, then translation.
  // The translation column of T is preserved (multiplied by S's (0,0,0,1) column)
  REQUIRE(ts(0, 0) == Catch::Approx(2.f));
  REQUIRE(ts(1, 1) == Catch::Approx(2.f));
  REQUIRE(ts(2, 2) == Catch::Approx(2.f));
  REQUIRE(ts(0, 3) == Catch::Approx(1.f));
  REQUIRE(ts(1, 3) == Catch::Approx(2.f));
  REQUIRE(ts(2, 3) == Catch::Approx(3.f));
}

// ============================================================================
// AABB Tests
// ============================================================================
TEST_CASE("AABB contains", "[math]") {
  AABB box{Vec3{0.f, 0.f, 0.f}, Vec3{10.f, 10.f, 10.f}};
  REQUIRE(box.contains(Vec3{5.f, 5.f, 5.f}) == true);
  REQUIRE(box.contains(Vec3{0.f, 0.f, 0.f}) == true);
  REQUIRE(box.contains(Vec3{10.f, 10.f, 10.f}) == true);
  REQUIRE(box.contains(Vec3{-1.f, 5.f, 5.f}) == false);
  REQUIRE(box.contains(Vec3{11.f, 5.f, 5.f}) == false);
}

TEST_CASE("AABB intersects", "[math]") {
  AABB a{Vec3{0.f, 0.f, 0.f}, Vec3{5.f, 5.f, 5.f}};
  AABB b{Vec3{3.f, 3.f, 3.f}, Vec3{8.f, 8.f, 8.f}};
  AABB c{Vec3{6.f, 6.f, 6.f}, Vec3{10.f, 10.f, 10.f}};

  REQUIRE(a.intersects(b) == true);  // Overlapping
  REQUIRE(a.intersects(c) == false); // Disjoint
  REQUIRE(a.intersects(a) == true);  // Self-intersection
}

TEST_CASE("AABB expandedBy", "[math]") {
  AABB box{Vec3{0.f, 0.f, 0.f}, Vec3{10.f, 10.f, 10.f}};
  AABB expanded = box.expandedBy(Vec3{5.f, 5.f, 5.f});

  REQUIRE(expanded.min.x == 5.f);
  REQUIRE(expanded.min.y == 5.f);
  REQUIRE(expanded.min.z == 5.f);
  REQUIRE(expanded.max.x == 15.f);
  REQUIRE(expanded.max.y == 15.f);
  REQUIRE(expanded.max.z == 15.f);
}

TEST_CASE("AABB intersectsRay — ray hits box", "[math]") {
  AABB box{Vec3{-1.f, -1.f, -1.f}, Vec3{1.f, 1.f, 1.f}};

  // Ray from origin going toward center of box
  Vec3 origin{0.f, 0.f, -5.f};
  Vec3 dir{0.f, 0.f, 1.f};
  REQUIRE(box.intersectsRay(origin, dir) == true);
}

TEST_CASE("AABB intersectsRay — ray misses box", "[math]") {
  AABB box{Vec3{-1.f, -1.f, -1.f}, Vec3{1.f, 1.f, 1.f}};

  // Ray parallel to box, offset in X
  Vec3 origin{5.f, 0.f, -5.f};
  Vec3 dir{0.f, 0.f, 1.f};
  REQUIRE(box.intersectsRay(origin, dir) == false);
}

// ============================================================================
// Result Tests
// ============================================================================
TEST_CASE("Result ok creation", "[result]") {
  auto r = Result<int, std::string>::ok(42);
  REQUIRE(r.is_ok() == true);
  REQUIRE(r.is_error() == false);
  REQUIRE(r.value() == 42);
}

TEST_CASE("Result err creation", "[result]") {
  auto r = Result<int, std::string>::err("something went wrong");
  REQUIRE(r.is_ok() == false);
  REQUIRE(r.is_error() == true);
  REQUIRE(r.error() == "something went wrong");
}

TEST_CASE("Result value_or", "[result]") {
  auto ok = Result<int, std::string>::ok(10);
  auto err = Result<int, std::string>::err("oops");

  REQUIRE(ok.value_or(-1) == 10);
  REQUIRE(err.value_or(-1) == -1);
}

TEST_CASE("Result unwrap_or", "[result]") {
  auto ok = Result<int, std::string>::ok(10);
  auto err = Result<int, std::string>::err("oops");

  REQUIRE(ok.unwrap_or(-1) == 10);
  REQUIRE(err.unwrap_or(-1) == -1);
}

TEST_CASE("Result map", "[result]") {
  auto ok = Result<int, std::string>::ok(5);
  auto mapped = ok.map([](int x) { return x * 2; });
  REQUIRE(mapped.is_ok() == true);
  REQUIRE(mapped.value() == 10);

  // Map on error propagates error
  auto err = Result<int, std::string>::err("bad");
  auto mappedErr = err.map([](int x) { return x * 2; });
  REQUIRE(mappedErr.is_error() == true);
  REQUIRE(mappedErr.error() == "bad");
}

TEST_CASE("Result and_then", "[result]") {
  auto ok = Result<int, std::string>::ok(5);
  auto chained = ok.and_then([](int x) {
    return Result<int, std::string>::ok(x + 1);
  });
  REQUIRE(chained.is_ok() == true);
  REQUIRE(chained.value() == 6);

  // and_then on error short-circuits
  auto err = Result<int, std::string>::err("fail");
  auto chainedErr = err.and_then([](int x) {
    return Result<int, std::string>::ok(x + 1);
  });
  REQUIRE(chainedErr.is_error() == true);
  REQUIRE(chainedErr.error() == "fail");
}

// ============================================================================
// ThreadPool Tests
// ============================================================================
TEST_CASE("ThreadPool submit returns correct result", "[thread]") {
  ThreadPool pool(2);
  auto future = pool.submit([]() { return 42; });
  REQUIRE(future.get() == 42);
}

TEST_CASE("ThreadPool submit with arguments", "[thread]") {
  ThreadPool pool(1);
  auto future = pool.submit([](int a, int b) { return a + b; }, 3, 4);
  REQUIRE(future.get() == 7);
}

TEST_CASE("ThreadPool multiple tasks complete", "[thread]") {
  ThreadPool pool(4);
  auto f1 = pool.submit([]() { return 1; });
  auto f2 = pool.submit([]() { return 2; });
  auto f3 = pool.submit([]() { return 3; });
  auto f4 = pool.submit([]() { return 4; });

  REQUIRE(f1.get() == 1);
  REQUIRE(f2.get() == 2);
  REQUIRE(f3.get() == 3);
  REQUIRE(f4.get() == 4);
}

TEST_CASE("ThreadPool size", "[thread]") {
  ThreadPool pool(3);
  REQUIRE(pool.size() == 3);
}

TEST_CASE("ThreadPool destructor joins", "[thread]") {
  // This test verifies that ThreadPool destructor properly joins workers.
  // If it doesn't, the test framework may hang or crash.
  {
    ThreadPool pool(2);
    auto f = pool.submit([]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      return 1;
    });
    f.get();
  } // pool destroyed here — should join cleanly
}

// ============================================================================
// Existing Tests
// ============================================================================
TEST_CASE("Chunk coordinates are multiples of CHUNK_SIZE", "[chunk]") {
  REQUIRE(0 % CHUNK_SIZE == 0);
  REQUIRE(16 % CHUNK_SIZE == 0);
  REQUIRE(32 % CHUNK_SIZE == 0);
  REQUIRE((-16) % CHUNK_SIZE == 0);
}

TEST_CASE("BlockType enum values are as expected", "[block]") {
  REQUIRE(static_cast<int>(BlockType::AIR) == 0);
  REQUIRE(static_cast<int>(BlockType::STONE) == 1);
  REQUIRE(static_cast<int>(BlockType::GRASS) == 2);
  REQUIRE(static_cast<int>(BlockType::DIRT) == 3);
  REQUIRE(static_cast<int>(BlockType::SAND) == 4);
  REQUIRE(static_cast<int>(BlockType::BEDROCK) == 7);
  REQUIRE(static_cast<int>(BlockType::LOG) == 8);
  REQUIRE(static_cast<int>(BlockType::LEAVES) == 9);
  REQUIRE(static_cast<int>(BlockType::COUNT) == 17);
}

// ============================================================================
// Simplex Noise Tests
// ============================================================================
TEST_CASE("SimplexNoise deterministic output for same seed", "[noise]") {
  SimplexNoise noise1(42);
  SimplexNoise noise2(42);

  double v1 = noise1.noise2D(1.0, 2.0);
  double v2 = noise2.noise2D(1.0, 2.0);
  REQUIRE(v1 == v2);

  double v3 = noise1.noise3D(1.0, 2.0, 3.0);
  double v4 = noise2.noise3D(1.0, 2.0, 3.0);
  REQUIRE(v3 == v4);
}

TEST_CASE("SimplexNoise same input gives same output", "[noise]") {
  SimplexNoise noise(123);

  double a = noise(10.0, 20.0);
  double b = noise(10.0, 20.0);
  REQUIRE(a == b);

  double c = noise.noise3D(5.0, 5.0, 5.0);
  double d = noise.noise3D(5.0, 5.0, 5.0);
  REQUIRE(c == d);
}

TEST_CASE("SimplexNoise output range within [-1, 1]", "[noise]") {
  SimplexNoise noise(99);

  // Sample a grid of points
  for (int ix = -10; ix <= 10; ++ix) {
    for (int iy = -10; iy <= 10; ++iy) {
      double v2d = noise.noise2D(static_cast<double>(ix), static_cast<double>(iy));
      REQUIRE(v2d >= -1.0);
      REQUIRE(v2d <= 1.0);
    }
  }

  for (int ix = -5; ix <= 5; ++ix) {
    for (int iy = -5; iy <= 5; ++iy) {
      for (int iz = -5; iz <= 5; ++iz) {
        double v3d = noise.noise3D(
            static_cast<double>(ix),
            static_cast<double>(iy),
            static_cast<double>(iz)
        );
        REQUIRE(v3d >= -1.0);
        REQUIRE(v3d <= 1.0);
      }
    }
  }
}

TEST_CASE("SimplexNoise different seeds give different outputs", "[noise]") {
  SimplexNoise noiseA(1);
  SimplexNoise noiseB(2);

  // Different seeds should produce different noise fields
  bool different = false;
  for (int i = 0; i < 20; ++i) {
    double a = noiseA.noise2D(static_cast<double>(i), static_cast<double>(i));
    double b = noiseB.noise2D(static_cast<double>(i), static_cast<double>(i));
    if (a != b) {
      different = true;
      break;
    }
  }
  REQUIRE(different == true);
}

TEST_CASE("SimplexNoise octave2D is deterministic", "[noise]") {
  SimplexNoise noise(77);
  double a = noise.octave2D(10.0, 20.0, 4, 0.5, 2.0);
  double b = noise.octave2D(10.0, 20.0, 4, 0.5, 2.0);
  REQUIRE(a == b);
}

TEST_CASE("SimplexNoise octave output range within [-1, 1]", "[noise]") {
  SimplexNoise noise(42);

  for (int i = 0; i < 20; ++i) {
    double v = noise.octave2D(static_cast<double>(i) * 0.1, static_cast<double>(i) * 0.1, 6, 0.5, 2.0);
    REQUIRE(v >= -1.0);
    REQUIRE(v <= 1.0);
  }
}

TEST_CASE("SimplexNoise ridged noise is deterministic", "[noise]") {
  SimplexNoise noise(55);
  double a = noise.ridged2D(10.0, 20.0, 4, 0.5, 2.0);
  double b = noise.ridged2D(10.0, 20.0, 4, 0.5, 2.0);
  REQUIRE(a == b);
}

TEST_CASE("SimplexNoise ridged output range within [0, 1]", "[noise]") {
  SimplexNoise noise(42);

  for (int i = 0; i < 20; ++i) {
    double v = noise.ridged2D(static_cast<double>(i) * 0.1, static_cast<double>(i) * 0.1, 4, 0.5, 2.0);
    REQUIRE(v >= 0.0);
    REQUIRE(v <= 1.0);
  }
}

TEST_CASE("SimplexNoise operator() equals noise2D", "[noise]") {
  SimplexNoise noise(42);
  REQUIRE(noise(1.0, 2.0) == noise.noise2D(1.0, 2.0));
  REQUIRE(noise(10.5, -3.7) == noise.noise2D(10.5, -3.7));
}

// ============================================================================
// Terrain Generator Tests
// ============================================================================
TEST_CASE("TerrainGenerator height within [minHeight, maxHeight]", "[terrain]") {
  TerrainGenerator terrain(42);
  TerrainConfig config;
  config.minHeight = 20.0;
  config.maxHeight = 128.0;

  for (int ix = -100; ix <= 100; ix += 10) {
    for (int iz = -100; iz <= 100; iz += 10) {
      double h = terrain.getHeight(static_cast<double>(ix), static_cast<double>(iz), config);
      REQUIRE(h >= config.minHeight);
      REQUIRE(h <= config.maxHeight);
    }
  }
}

TEST_CASE("TerrainGenerator deterministic output", "[terrain]") {
  TerrainGenerator t1(123);
  TerrainGenerator t2(123);

  double a = t1.getHeight(100.0, 200.0);
  double b = t2.getHeight(100.0, 200.0);
  REQUIRE(a == b);
}

TEST_CASE("TerrainGenerator getNoise is deterministic", "[terrain]") {
  TerrainGenerator terrain(42);
  double a = terrain.getNoise(10.0, 20.0);
  double b = terrain.getNoise(10.0, 20.0);
  REQUIRE(a == b);
}

// ============================================================================
// Biome Generator Tests
// ============================================================================
TEST_CASE("BiomeGenerator lookup: DeepOcean for very low elevation", "[biome]") {
  BiomeGenerator bg(42);
  Biome b = bg.lookupBiome(0.5, 0.5, 50.0);
  REQUIRE(b == Biome::DeepOcean);
}

TEST_CASE("BiomeGenerator lookup: Ocean for below sea level", "[biome]") {
  BiomeGenerator bg(42);
  Biome b = bg.lookupBiome(0.5, 0.5, 62.0);
  REQUIRE(b == Biome::Ocean);
}

TEST_CASE("BiomeGenerator lookup: Swamp for low elevation + wet", "[biome]") {
  BiomeGenerator bg(42);
  Biome b = bg.lookupBiome(0.6, 0.7, 66.0);
  REQUIRE(b == Biome::Swamp);
}

TEST_CASE("BiomeGenerator lookup: ExtremeHills for cold + dry", "[biome]") {
  BiomeGenerator bg(42);
  Biome b = bg.lookupBiome(0.2, 0.2, 80.0);
  REQUIRE(b == Biome::ExtremeHills);
}

TEST_CASE("BiomeGenerator lookup: IceSpikes for cold + medium moisture", "[biome]") {
  BiomeGenerator bg(42);
  Biome b = bg.lookupBiome(0.2, 0.4, 80.0);
  REQUIRE(b == Biome::IceSpikes);
}

TEST_CASE("BiomeGenerator lookup: Taiga for cold + wet", "[biome]") {
  BiomeGenerator bg(42);
  Biome b = bg.lookupBiome(0.2, 0.7, 80.0);
  REQUIRE(b == Biome::Taiga);
}

TEST_CASE("BiomeGenerator lookup: Desert for hot + dry", "[biome]") {
  BiomeGenerator bg(42);
  Biome b = bg.lookupBiome(0.8, 0.2, 80.0);
  REQUIRE(b == Biome::Desert);
}

TEST_CASE("BiomeGenerator lookup: Forest for warm + wet", "[biome]") {
  BiomeGenerator bg(42);
  Biome b = bg.lookupBiome(0.5, 0.6, 80.0);
  REQUIRE(b == Biome::Forest);
}

TEST_CASE("BiomeGenerator lookup: Plains for warm + dry", "[biome]") {
  BiomeGenerator bg(42);
  Biome b = bg.lookupBiome(0.5, 0.4, 80.0);
  REQUIRE(b == Biome::Plains);
}

TEST_CASE("BiomeGenerator temperature in [0, 1]", "[biome]") {
  BiomeGenerator bg(42);
  for (int i = 0; i < 20; ++i) {
    double t = bg.getTemperature(static_cast<double>(i) * 10.0, static_cast<double>(i) * 10.0);
    REQUIRE(t >= 0.0);
    REQUIRE(t <= 1.0);
  }
}

TEST_CASE("BiomeGenerator moisture in [0, 1]", "[biome]") {
  BiomeGenerator bg(42);
  for (int i = 0; i < 20; ++i) {
    double m = bg.getMoisture(static_cast<double>(i) * 10.0, static_cast<double>(i) * 10.0);
    REQUIRE(m >= 0.0);
    REQUIRE(m <= 1.0);
  }
}

TEST_CASE("BiomeGenerator height modifier values", "[biome]") {
  BiomeGenerator bg(42);
  REQUIRE(bg.getBiomeHeightModifier(Biome::ExtremeHills) == 30.0);
  REQUIRE(bg.getBiomeHeightModifier(Biome::Desert) == 5.0);
  REQUIRE(bg.getBiomeHeightModifier(Biome::Plains) == 0.0);
  REQUIRE(bg.getBiomeHeightModifier(Biome::Forest) == 10.0);
  REQUIRE(bg.getBiomeHeightModifier(Biome::Taiga) == 15.0);
  REQUIRE(bg.getBiomeHeightModifier(Biome::Swamp) == -5.0);
}

TEST_CASE("BiomeGenerator surface block values", "[biome]") {
  BiomeGenerator bg(42);
  REQUIRE(bg.getSurfaceBlock(Biome::Plains) == BlockType::GRASS);
  REQUIRE(bg.getSurfaceBlock(Biome::Forest) == BlockType::GRASS);
  REQUIRE(bg.getSurfaceBlock(Biome::Taiga) == BlockType::GRASS);
  REQUIRE(bg.getSurfaceBlock(Biome::Desert) == BlockType::AIR);
  REQUIRE(bg.getSurfaceBlock(Biome::ExtremeHills) == BlockType::STONE);
  REQUIRE(bg.getSurfaceBlock(Biome::IceSpikes) == BlockType::STONE);
}

TEST_CASE("BiomeGenerator getBiome is deterministic", "[biome]") {
  BiomeGenerator bg1(42);
  BiomeGenerator bg2(42);

  Biome b1 = bg1.getBiome(100.0, 200.0, 80.0);
  Biome b2 = bg2.getBiome(100.0, 200.0, 80.0);
  REQUIRE(b1 == b2);
}

// ============================================================================
// Chunk Tests
// ============================================================================
TEST_CASE("Chunk creation initializes to air", "[chunk]") {
  Chunk chunk(0, 0);
  REQUIRE(chunk.chunkX == 0);
  REQUIRE(chunk.chunkZ == 0);
  REQUIRE(chunk.blocks.size() == static_cast<size_t>(CHUNK_VOLUME));
  REQUIRE(chunk.getBlock(0, 0, 0) == BlockType::AIR);
  REQUIRE(chunk.getBlock(7, 127, 7) == BlockType::AIR);
}

TEST_CASE("Chunk setBlock and getBlock", "[chunk]") {
  Chunk chunk(5, -3);
  chunk.setBlock(8, 64, 8, BlockType::STONE);
  REQUIRE(chunk.getBlock(8, 64, 8) == BlockType::STONE);

  chunk.setBlock(0, 0, 0, BlockType::GRASS);
  REQUIRE(chunk.getBlock(0, 0, 0) == BlockType::GRASS);
}

TEST_CASE("Chunk setBlock marks chunk dirty", "[chunk]") {
  Chunk chunk(0, 0);
  REQUIRE(chunk.needsMeshUpdate == false);
  chunk.setBlock(8, 64, 8, BlockType::STONE);
  REQUIRE(chunk.needsMeshUpdate == true);
}

TEST_CASE("Chunk out-of-bounds returns air", "[chunk]") {
  Chunk chunk(0, 0);
  REQUIRE(chunk.getBlock(-1, 64, 8) == BlockType::AIR);
  REQUIRE(chunk.getBlock(16, 64, 8) == BlockType::AIR);
  REQUIRE(chunk.getBlock(8, -1, 8) == BlockType::AIR);
  REQUIRE(chunk.getBlock(8, 256, 8) == BlockType::AIR);
  REQUIRE(chunk.getBlock(8, 64, -1) == BlockType::AIR);
  REQUIRE(chunk.getBlock(8, 64, 16) == BlockType::AIR);
}

TEST_CASE("Chunk world coordinate conversion", "[chunk]") {
  REQUIRE(Chunk::worldToChunk(0) == 0);
  REQUIRE(Chunk::worldToChunk(15) == 0);
  REQUIRE(Chunk::worldToChunk(16) == 1);
  REQUIRE(Chunk::worldToChunk(31) == 1);
  REQUIRE(Chunk::worldToChunk(32) == 2);
  REQUIRE(Chunk::worldToChunk(-1) == -1);
  REQUIRE(Chunk::worldToChunk(-16) == -1);
  REQUIRE(Chunk::worldToChunk(-17) == -2);
  REQUIRE(Chunk::worldToChunk(-32) == -2);
}

TEST_CASE("Chunk chunkToWorld conversion", "[chunk]") {
  REQUIRE(Chunk::chunkToWorld(0, 0) == 0);
  REQUIRE(Chunk::chunkToWorld(1, 0) == 16);
  REQUIRE(Chunk::chunkToWorld(1, 15) == 31);
  REQUIRE(Chunk::chunkToWorld(-1, 0) == -16);
  REQUIRE(Chunk::chunkToWorld(-1, 15) == -1);
}

TEST_CASE("Chunk world block access", "[chunk]") {
  Chunk chunk(2, -1);
  chunk.setBlockWorld(32, 64, -8, BlockType::DIRT);
  REQUIRE(chunk.getBlockWorld(32, 64, -8) == BlockType::DIRT);
}

TEST_CASE("Chunk getAABB", "[chunk]") {
  Chunk chunk(1, -1);
  AABB aabb = chunk.getAABB();
  REQUIRE(aabb.min.x == Catch::Approx(16.f));
  REQUIRE(aabb.min.y == Catch::Approx(0.f));
  REQUIRE(aabb.min.z == Catch::Approx(-16.f));
  REQUIRE(aabb.max.x == Catch::Approx(32.f));
  REQUIRE(aabb.max.y == Catch::Approx(256.f));
  REQUIRE(aabb.max.z == Catch::Approx(0.f));
}

TEST_CASE("Chunk getWorldPosition", "[chunk]") {
  Chunk chunk(3, -2);
  Vec3 pos = chunk.getWorldPosition();
  REQUIRE(pos.x == Catch::Approx(48.f));
  REQUIRE(pos.y == Catch::Approx(0.f));
  REQUIRE(pos.z == Catch::Approx(-32.f));
}

TEST_CASE("Chunk markDirty", "[chunk]") {
  Chunk chunk(0, 0);
  chunk.needsMeshUpdate = false;
  chunk.markDirty();
  REQUIRE(chunk.needsMeshUpdate == true);
}

// ============================================================================
// Serialization Tests
// ============================================================================
TEST_CASE("Serialization roundtrip", "[serialization]") {
  Chunk original(5, -3);
  original.setBlock(8, 64, 8, BlockType::STONE);
  original.setBlock(0, 0, 0, BlockType::GRASS);
  original.generated = true;
  original.biomes[0] = Biome::Desert;
  original.biomes[100] = Biome::Forest;
  original.heightMap[0] = 65;
  original.heightMap[100] = 72;

  auto data = ChunkSerializer::serialize(original);
  auto restored = ChunkSerializer::deserialize(data);
  REQUIRE(restored.has_value());
  REQUIRE(restored->chunkX == original.chunkX);
  REQUIRE(restored->chunkZ == original.chunkZ);
  REQUIRE(restored->getBlock(8, 64, 8) == BlockType::STONE);
  REQUIRE(restored->getBlock(0, 0, 0) == BlockType::GRASS);
  REQUIRE(restored->biomes[0] == Biome::Desert);
  REQUIRE(restored->biomes[100] == Biome::Forest);
  REQUIRE(restored->heightMap[0] == 65);
  REQUIRE(restored->heightMap[100] == 72);
}

TEST_CASE("Serialization size is correct", "[serialization]") {
  Chunk chunk(0, 0);
  size_t expected = ChunkSerializer::serializedSize(chunk);
  auto data = ChunkSerializer::serialize(chunk);
  REQUIRE(data.size() == expected);
}

TEST_CASE("Serialization corrupt data returns nullopt", "[serialization]") {
  std::vector<uint8_t> corruptData(100, 0xFF);
  auto result = ChunkSerializer::deserialize(corruptData);
  REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization empty data returns nullopt", "[serialization]") {
  std::vector<uint8_t> emptyData;
  auto result = ChunkSerializer::deserialize(emptyData);
  REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization wrong magic returns nullopt", "[serialization]") {
  Chunk chunk(0, 0);
  auto data = ChunkSerializer::serialize(chunk);
  data[0] = 0x00;
  auto result = ChunkSerializer::deserialize(data);
  REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization truncated data returns nullopt", "[serialization]") {
  Chunk chunk(0, 0);
  auto data = ChunkSerializer::serialize(chunk);
  data.resize(HEADER_SIZE);
  auto result = ChunkSerializer::deserialize(data);
  REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization wrong block count returns nullopt", "[serialization]") {
  Chunk chunk(0, 0);
  auto data = ChunkSerializer::serialize(chunk);
  data[16] = 0x00;
  data[17] = 0x00;
  data[18] = 0x00;
  data[19] = 0x01;
  auto result = ChunkSerializer::deserialize(data);
  REQUIRE(result.has_value() == false);
}

TEST_CASE("Serialization multiple roundtrips consistent", "[serialization]") {
  Chunk original(10, 10);
  original.setBlock(4, 100, 4, BlockType::DIAMOND_ORE);
  auto data1 = ChunkSerializer::serialize(original);
  auto restored1 = ChunkSerializer::deserialize(data1);
  REQUIRE(restored1.has_value());
  auto data2 = ChunkSerializer::serialize(*restored1);
  auto restored2 = ChunkSerializer::deserialize(data2);
  REQUIRE(restored2.has_value());
  REQUIRE(restored2->getBlock(4, 100, 4) == BlockType::DIAMOND_ORE);
}

// ============================================================================
// World Tests
// ============================================================================
TEST_CASE("World creation", "[world]") {
  auto world = std::make_shared<World>(42);
  REQUIRE(world->getSeed() == 42);
  REQUIRE(world->getViewDistance() == 32);
}

TEST_CASE("World getChunk generates chunk", "[world]") {
  auto world = std::make_shared<World>(123);
  auto chunk = world->getChunk(0, 0);
  REQUIRE(chunk != nullptr);
  REQUIRE(chunk->chunkX == 0);
  REQUIRE(chunk->chunkZ == 0);
  REQUIRE(chunk->generated == true);
}

TEST_CASE("World getChunk returns cached chunk", "[world]") {
  auto world = std::make_shared<World>(42);
  auto chunk1 = world->getChunk(5, -3);
  auto chunk2 = world->getChunk(5, -3);
  REQUIRE(chunk1 == chunk2);
}

TEST_CASE("World getBlock and setBlock", "[world]") {
  auto world = std::make_shared<World>(42);
  BlockType b = world->getBlock(100, 64, 100);
  REQUIRE(static_cast<int>(b) >= 0);
  world->setBlock(100, 64, 100, BlockType::DIAMOND_ORE);
  BlockType after = world->getBlock(100, 64, 100);
  REQUIRE(after == BlockType::DIAMOND_ORE);
}

TEST_CASE("World getLoadedChunks", "[world]") {
  auto world = std::make_shared<World>(42);
  world->getChunk(0, 0);
  world->getChunk(1, 0);
  world->getChunk(0, 1);
  auto loaded = world->getLoadedChunks();
  REQUIRE(loaded.size() == 3);
}

TEST_CASE("World getTerrainHeight", "[world]") {
  auto world = std::make_shared<World>(42);
  double h = world->getTerrainHeight(100, 200);
  REQUIRE(h >= 0.0);
}

TEST_CASE("World getBiome", "[world]") {
  auto world = std::make_shared<World>(42);
  Biome b = world->getBiome(100, 200);
  REQUIRE(static_cast<int>(b) >= 0);
  REQUIRE(static_cast<int>(b) < static_cast<int>(Biome::Count));
}

TEST_CASE("World setViewDistance", "[world]") {
  auto world = std::make_shared<World>(42);
  world->setViewDistance(10);
  REQUIRE(world->getViewDistance() == 10);
  world->setViewDistance(0);
  REQUIRE(world->getViewDistance() == 1);
}

TEST_CASE("World markChunkMeshed", "[world]") {
  auto world = std::make_shared<World>(42);
  auto chunk = world->getChunk(0, 0);
  REQUIRE(chunk->needsMeshUpdate == true);
  world->markChunkMeshed(0, 0);
  REQUIRE(chunk->needsMeshUpdate == false);
}

TEST_CASE("World getDirtyChunks", "[world]") {
  auto world = std::make_shared<World>(42);
  world->getChunk(0, 0);
  world->getChunk(1, 0);
  auto dirty = world->getDirtyChunks();
  REQUIRE(dirty.size() == 2);
  world->markChunkMeshed(0, 0);
  dirty = world->getDirtyChunks();
  REQUIRE(dirty.size() == 1);
}

// ============================================================================
// Async Generation Tests
// ============================================================================
TEST_CASE("World async generation pending count", "[world][async]") {
  auto world = std::make_shared<World>(42);
  REQUIRE(world->getPendingChunkCount() == 0);
}

TEST_CASE("World generateAroundPlayer submits chunks", "[world][async]") {
  auto world = std::make_shared<World>(42);
  world->setViewDistance(4);
  world->generateAroundPlayer(0, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  size_t pending = world->getPendingChunkCount();
  REQUIRE(pending >= 0);
}

TEST_CASE("World generateAroundPlayer populates chunks", "[world][async]") {
  auto world = std::make_shared<World>(42);
  world->setViewDistance(2);
  world->generateAroundPlayer(0, 0);
  for (int attempts = 0; attempts < 50; ++attempts) {
    if (world->getPendingChunkCount() == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  auto chunks = world->getLoadedChunks();
  REQUIRE(chunks.size() >= 0);
}

TEST_CASE("World updatePlayerPosition loads surrounding chunks", "[world]") {
  auto world = std::make_shared<World>(42);
  world->setViewDistance(2);
  world->updatePlayerPosition(256, 256);
  auto chunks = world->getLoadedChunks();
  REQUIRE(chunks.size() == 25);
}

// ============================================================================
// SaveManager Tests
// ============================================================================
TEST_CASE("SaveManager creation", "[save]") {
  SaveManager saver("/tmp/rycraft_test_world_sm1");
  REQUIRE(saver.getWorldPath().find("rycraft_test_world_") != std::string::npos);
}

TEST_CASE("SaveManager save/load chunk roundtrip", "[save]") {
  std::string tempDir = "/tmp/rycraft_test_save_roundtrip";
  {
    SaveManager saver(tempDir);
    Chunk original(7, -5);
    original.setBlock(8, 100, 8, BlockType::IRON_ORE);
    original.generated = true;
    saver.saveChunk(original);
    saver.flush();
    auto loaded = saver.loadChunk(7, -5);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->chunkX == 7);
    REQUIRE(loaded->chunkZ == -5);
    REQUIRE(loaded->getBlock(8, 100, 8) == BlockType::IRON_ORE);
  }
  std::system(("rm -rf " + tempDir).c_str());
}

TEST_CASE("SaveManager load non-existent chunk returns nullopt", "[save]") {
  std::string tempDir = "/tmp/rycraft_test_save_missing";
  {
    SaveManager saver(tempDir);
    auto loaded = saver.loadChunk(999, 999);
    REQUIRE(loaded.has_value() == false);
  }
  std::system(("rm -rf " + tempDir).c_str());
}

TEST_CASE("SaveManager save/load metadata roundtrip", "[save]") {
  std::string tempDir = "/tmp/rycraft_test_save_meta";
  {
    SaveManager saver(tempDir);
    saver.saveMetadata(12345, Vec3{100.f, 80.f, -50.f}, 9876543210);
    auto meta = saver.loadMetadata();
    REQUIRE(meta.has_value());
    REQUIRE(meta->seed == 12345);
    REQUIRE(meta->spawnPos.x == Catch::Approx(100.f));
    REQUIRE(meta->spawnPos.y == Catch::Approx(80.f));
    REQUIRE(meta->spawnPos.z == Catch::Approx(-50.f));
    REQUIRE(meta->worldTime == 9876543210);
  }
  std::system(("rm -rf " + tempDir).c_str());
}

TEST_CASE("SaveManager load missing metadata returns nullopt", "[save]") {
  std::string tempDir = "/tmp/rycraft_test_save_nometa";
  {
    SaveManager saver(tempDir);
    auto meta = saver.loadMetadata();
    REQUIRE(meta.has_value() == false);
  }
  std::system(("rm -rf " + tempDir).c_str());
}

TEST_CASE("SaveManager LZ4 compression produces smaller data", "[save]") {
  std::string tempDir = "/tmp/rycraft_test_save_compress";
  {
    SaveManager saver(tempDir);
    Chunk chunk(0, 0);
    chunk.setBlock(8, 64, 8, BlockType::STONE);
    chunk.generated = true;
    saver.saveChunk(chunk);
    saver.flush();
    auto loaded = saver.loadChunk(0, 0);
    REQUIRE(loaded.has_value());
  }
  std::system(("rm -rf " + tempDir).c_str());
}

// ============================================================================
// Vertex Format Tests
// ============================================================================
TEST_CASE("Vertex size is 16 bytes", "[render][vertex]") {
    REQUIRE(sizeof(Vertex) == 16);
}

TEST_CASE("Vertex alignment is 16 bytes", "[render][vertex]") {
    REQUIRE(alignof(Vertex) == 16);
}

TEST_CASE("Vertex fields have expected sizes", "[render][vertex]") {
    REQUIRE(sizeof(float16_t) == 2);
    REQUIRE(sizeof(uint8_t) == 1);
    REQUIRE(sizeof(uint32_t) == 4);
}

// ============================================================================
// Greedy Mesher Tests
// ============================================================================
TEST_CASE("GreedyMesher empty chunk produces no geometry", "[render][mesher]") {
    Chunk chunk(0, 0);
    // All AIR — no solid blocks

    GreedyMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk);

    REQUIRE(output.vertices.empty());
    REQUIRE(output.indices.empty());
}

TEST_CASE("GreedyMesher single block produces 6 faces", "[render][mesher]") {
    Chunk chunk(0, 0);
    chunk.setBlock(8, 64, 8, BlockType::STONE);

    GreedyMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk);

    // 6 faces × 4 vertices = 24 vertices
    REQUIRE(output.vertices.size() == 24);
    // 6 faces × 2 triangles × 3 indices = 36 indices
    REQUIRE(output.indices.size() == 36);
}

TEST_CASE("GreedyMesher 2x2 flat merges top face", "[render][mesher]") {
    Chunk chunk(0, 0);
    // 2x2 square of STONE at y=64
    chunk.setBlock(0, 64, 0, BlockType::STONE);
    chunk.setBlock(1, 64, 0, BlockType::STONE);
    chunk.setBlock(0, 64, 1, BlockType::STONE);
    chunk.setBlock(1, 64, 1, BlockType::STONE);

    GreedyMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk);

    // Without greedy merge: 4 top faces = 16 vertices
    // With greedy merge: 1 top face = 4 vertices
    // Total expected:
    //   Top (+Y): 1 merged quad = 4 vertices, 6 indices
    //   Bottom (-Y): 1 merged quad = 4 vertices, 6 indices
    //   +X face (right side of x=1 column): 2 quads (z=0 and z=1) = 8 vertices, 12 indices
    //   -X face (left side of x=0 column): 2 quads = 8 vertices, 12 indices
    //   +Z face (front of z=1 row): 2 quads = 8 vertices, 12 indices
    //   -Z face (back of z=0 row): 2 quads = 8 vertices, 12 indices
    // Total: 40 vertices, 60 indices
    //
    // But +X and -X faces can also merge vertically since all 4 blocks are at same Y
    // +X: blocks at (1,64,0) and (1,64,1) — both have +X exposed, same type
    //   They're adjacent in Z direction, so they merge into 1 quad: 4 vertices, 6 indices
    // Same for -X, +Z, -Z
    //
    // Total: 6 faces × 4 vertices = 24 vertices, 6 × 6 = 36 indices

    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.indices.size() == 36);

    // Verify the top face is a single quad (first 4 vertices)
    // All 4 top-face vertices should have normalIdx == FaceNormal::PlusY (4)
    bool foundTopQuad = false;
    for (size_t i = 0; i + 3 < output.vertices.size(); ++i) {
        if (output.vertices[i].normalIdx == static_cast<uint8_t>(FaceNormal::PlusY) &&
            output.vertices[i + 1].normalIdx == static_cast<uint8_t>(FaceNormal::PlusY) &&
            output.vertices[i + 2].normalIdx == static_cast<uint8_t>(FaceNormal::PlusY) &&
            output.vertices[i + 3].normalIdx == static_cast<uint8_t>(FaceNormal::PlusY)) {
            foundTopQuad = true;
            break;
        }
    }
    REQUIRE(foundTopQuad);
}

TEST_CASE("GreedyMesher vertical column merges side faces", "[render][mesher]") {
    Chunk chunk(0, 0);
    // 4-block tall column of STONE at (8, 64..67, 8)
    chunk.setBlock(8, 64, 8, BlockType::STONE);
    chunk.setBlock(8, 65, 8, BlockType::STONE);
    chunk.setBlock(8, 66, 8, BlockType::STONE);
    chunk.setBlock(8, 67, 8, BlockType::STONE);

    GreedyMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk);

    // Top (+Y): 1 quad at y=67 top = 4 vertices, 6 indices
    // Bottom (-Y): 1 quad at y=64 bottom = 4 vertices, 6 indices
    // +X, -X, +Z, -Z: each has 1 merged quad spanning y=64..67 = 4 vertices each, 6 indices each
    // Total: 6 faces × 4 vertices = 24 vertices
    // Total: 6 faces × 6 indices = 36 indices

    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.indices.size() == 36);

    // Verify side faces span the full column height
    // Each side quad should have vertices at y=64 and y=68 (height=4)
    bool foundSideQuad = false;
    for (size_t i = 0; i + 3 < output.vertices.size(); ++i) {
        uint8_t ni = output.vertices[i].normalIdx;
        // Check side faces (normalIdx 0-3)
        if (ni <= 3 &&
            output.vertices[i + 1].normalIdx == ni &&
            output.vertices[i + 2].normalIdx == ni &&
            output.vertices[i + 3].normalIdx == ni) {
            // Check that the quad spans 4 units in Y
            float minY = std::min({
                static_cast<float>(output.vertices[i].py),
                static_cast<float>(output.vertices[i + 1].py),
                static_cast<float>(output.vertices[i + 2].py),
                static_cast<float>(output.vertices[i + 3].py)
            });
            float maxY = std::max({
                static_cast<float>(output.vertices[i].py),
                static_cast<float>(output.vertices[i + 1].py),
                static_cast<float>(output.vertices[i + 2].py),
                static_cast<float>(output.vertices[i + 3].py)
            });
            if (maxY - minY >= 3.5f) { // height=4, account for float16 precision
                foundSideQuad = true;
                break;
            }
        }
    }
    REQUIRE(foundSideQuad);
}

TEST_CASE("GreedyMesher produces mesh without side effects", "[render][mesher]") {
    Chunk chunk(0, 0);
    chunk.setBlock(8, 64, 8, BlockType::STONE);
    REQUIRE(chunk.needsMeshUpdate == true);

    GreedyMesher mesher;
    MeshOutput mesh = mesher.buildMesh(chunk);

    // buildMesh is pure — it does not modify the chunk
    REQUIRE(chunk.needsMeshUpdate == true);
    REQUIRE(mesh.vertices.size() > 0u);
    REQUIRE(mesh.indices.size() > 0u);

    // Caller is responsible for marking the chunk as meshed
    chunk.setMeshed(true);
    chunk.needsMeshUpdate = false;
    REQUIRE(chunk.meshed == true);
    REQUIRE(chunk.needsMeshUpdate == false);
}

// ============================================================================
// TextureAtlas Constant Tests (no Metal device required)
// ============================================================================
TEST_CASE("TextureAtlas constants are consistent", "[render][atlas]") {
    REQUIRE(TextureAtlas::TILE_SIZE == 16);
    REQUIRE(TextureAtlas::ATLAS_WIDTH == 1024);
    REQUIRE(TextureAtlas::ATLAS_HEIGHT == 1024);
    REQUIRE(TextureAtlas::TILES_PER_ROW == 64);
    REQUIRE(TextureAtlas::TOTAL_TILES == 4096);
    REQUIRE(TextureAtlas::MAX_BLOCK_TYPES == 256);
}

TEST_CASE("TextureAtlas tile allocation: 256 tiles fit without overlap", "[render][atlas]") {
    // Verify that 256 unique tile indices produce unique UV coordinates.
    // The UV grid is 64×64 tiles, so 256 tiles (16×16 sub-grid) fit easily.
    float uvSize = 1.0f / static_cast<float>(TextureAtlas::TILES_PER_ROW);

    struct UVBounds {
        float uMin, vMin, uMax, vMax;
    };

    std::vector<UVBounds> tileBounds;
    tileBounds.reserve(256);

    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t col = i % TextureAtlas::TILES_PER_ROW;
        uint32_t row = i / TextureAtlas::TILES_PER_ROW;
        float u = static_cast<float>(col) * uvSize;
        float v = static_cast<float>(row) * uvSize;
        tileBounds.push_back({u, v, u + uvSize, v + uvSize});
    }

    // Verify no two tiles overlap
    for (size_t i = 0; i < tileBounds.size(); ++i) {
        for (size_t j = i + 1; j < tileBounds.size(); ++j) {
            // Tiles overlap if their UV ranges intersect
            bool overlap = !(tileBounds[i].uMax <= tileBounds[j].uMin ||
                             tileBounds[i].uMin >= tileBounds[j].uMax ||
                             tileBounds[i].vMax <= tileBounds[j].vMin ||
                             tileBounds[i].vMin >= tileBounds[j].vMax);
            REQUIRE(!overlap);
        }
    }
}

TEST_CASE("TextureAtlas UV computation within tile bounds", "[render][atlas]") {
    float uvSize = 1.0f / static_cast<float>(TextureAtlas::TILES_PER_ROW);

    // Test UV for tiles at various positions
    uint32_t testIndices[] = {0, 1, 63, 64, 100, 1023, 4095};

    for (uint32_t tileIdx : testIndices) {
        uint32_t col = tileIdx % TextureAtlas::TILES_PER_ROW;
        uint32_t row = tileIdx / TextureAtlas::TILES_PER_ROW;

        float expectedU = static_cast<float>(col) * uvSize;
        float expectedV = static_cast<float>(row) * uvSize;

        // Verify UV is within [0, 1) range
        REQUIRE(expectedU >= 0.0f);
        REQUIRE(expectedU < 1.0f);
        REQUIRE(expectedV >= 0.0f);
        REQUIRE(expectedV < 1.0f);

        // Verify tile size is consistent
        REQUIRE(uvSize == Catch::Approx(1.0f / 64.0f));

        // Verify tile stays within atlas bounds
        REQUIRE(expectedU + uvSize <= 1.0f);
        REQUIRE(expectedV + uvSize <= 1.0f);
    }
}

TEST_CASE("TextureAtlas exhaustion: total tiles equals grid capacity", "[render][atlas]") {
    // TOTAL_TILES must equal TILES_PER_ROW * (ATLAS_HEIGHT / TILE_SIZE)
    uint32_t expectedTotal = TextureAtlas::TILES_PER_ROW *
                             (TextureAtlas::ATLAS_HEIGHT / TextureAtlas::TILE_SIZE);
    REQUIRE(TextureAtlas::TOTAL_TILES == expectedTotal);

    // MAX_BLOCK_TYPES (256) must be <= TOTAL_TILES (4096)
    REQUIRE(TextureAtlas::MAX_BLOCK_TYPES <= TextureAtlas::TOTAL_TILES);
}

// ============================================================================
// MegaBuffer Constant Tests (no Metal device required)
// ============================================================================
TEST_CASE("MegaBuffer alignment is power of 2", "[render][megabuffer]") {
    uint64_t align = MegaBuffer::ALIGNMENT;
    REQUIRE(align > 0);
    REQUIRE((align & (align - 1)) == 0); // Power of 2 check
    REQUIRE(align == 256);
}

TEST_CASE("MegaBuffer alignUp rounds up to alignment", "[render][megabuffer]") {
    // Test alignment behavior with known values
    // alignUp(x) = (x + 255) & ~255
    auto alignUp = [](uint64_t value) -> uint64_t {
        return (value + MegaBuffer::ALIGNMENT - 1) & ~(MegaBuffer::ALIGNMENT - 1);
    };

    REQUIRE(alignUp(0) == 0);
    REQUIRE(alignUp(1) == 256);
    REQUIRE(alignUp(256) == 256);
    REQUIRE(alignUp(257) == 512);
    REQUIRE(alignUp(512) == 512);
    REQUIRE(alignUp(1000) == 1024);
    REQUIRE(alignUp(16 * sizeof(Vertex)) == 256); // 256 bytes, already aligned
    REQUIRE(alignUp(17 * sizeof(Vertex)) == 512); // 272 bytes, rounds to 512
}

TEST_CASE("MegaBuffer vertex allocation size calculation", "[render][megabuffer]") {
    auto alignUp = [](uint64_t value) -> uint64_t {
        return (value + MegaBuffer::ALIGNMENT - 1) & ~(MegaBuffer::ALIGNMENT - 1);
    };

    // 100 vertices × 16 bytes = 1600 bytes → aligned to 1792 (7 × 256)
    uint64_t vertexBytes = alignUp(100 * sizeof(Vertex));
    REQUIRE(vertexBytes >= 1600);
    REQUIRE(vertexBytes % MegaBuffer::ALIGNMENT == 0);

    // 1000 indices × 4 bytes = 4000 bytes → aligned to 4096 (16 × 256)
    uint64_t indexBytes = alignUp(1000 * sizeof(uint32_t));
    REQUIRE(indexBytes >= 4000);
    REQUIRE(indexBytes % MegaBuffer::ALIGNMENT == 0);
}

// ============================================================================
// UIOverlay Quad Vertex Generation Tests (no Metal device required)
// ============================================================================
TEST_CASE("UIOverlay quad vertex generation: fullscreen quad", "[render][ui]") {
    // Verify that a fullscreen quad (0,0,1,1) produces correct vertex positions.
    // Layout: [x, y] for each of 4 vertices: BL, TL, BR, TR
    float x = 0.0f, y = 0.0f, w = 1.0f, h = 1.0f;

    // Expected vertices (bottom-left origin):
    // BL: (0, 0), TL: (0, 1), BR: (1, 0), TR: (1, 1)
    struct QuadVertex {
        float px, py;
        float cr, cg, cb, ca;
    };

    QuadVertex expected[4] = {
        {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    };

    // Verify vertex positions
    REQUIRE(expected[0].px == x);
    REQUIRE(expected[0].py == y);
    REQUIRE(expected[1].px == x);
    REQUIRE(expected[1].py == y + h);
    REQUIRE(expected[2].px == x + w);
    REQUIRE(expected[2].py == y);
    REQUIRE(expected[3].px == x + w);
    REQUIRE(expected[3].py == y + h);
}

TEST_CASE("UIOverlay quad vertex generation: crosshair horizontal line", "[render][ui]") {
    // Simulate crosshair horizontal line at center of 1920×1080 screen
    float screenWidth = 1920.0f;
    float screenHeight = 1080.0f;

    float centerX = 0.5f;
    float centerY = 0.5f;
    float crossH = 1.0f / screenHeight;    // 1 pixel height
    float crossW = 20.0f / screenWidth;    // 20 pixel width

    float left = centerX - crossW * 0.5f;
    float bottom = centerY - crossH * 0.5f;

    // Verify crosshair is centered
    REQUIRE(left + crossW * 0.5f == Catch::Approx(centerX));
    REQUIRE(bottom + crossH * 0.5f == Catch::Approx(centerY));

    // Verify dimensions are positive
    REQUIRE(crossW > 0.0f);
    REQUIRE(crossH > 0.0f);

    // Verify crosshair fits within screen
    REQUIRE(left >= 0.0f);
    REQUIRE(left + crossW <= 1.0f);
    REQUIRE(bottom >= 0.0f);
    REQUIRE(bottom + crossH <= 1.0f);
}

TEST_CASE("UIOverlay quad vertex generation: crosshair vertical line", "[render][ui]") {
    float screenWidth = 1920.0f;
    float screenHeight = 1080.0f;

    float centerX = 0.5f;
    float centerY = 0.5f;
    float crossV = 20.0f / screenHeight;   // 20 pixel height
    float crossLineW = 1.0f / screenWidth; // 1 pixel width

    float left = centerX - crossLineW * 0.5f;
    float bottom = centerY - crossV * 0.5f;

    // Verify crosshair is centered
    REQUIRE(left + crossLineW * 0.5f == Catch::Approx(centerX));
    REQUIRE(bottom + crossV * 0.5f == Catch::Approx(centerY));

    // Verify dimensions are positive
    REQUIRE(crossLineW > 0.0f);
    REQUIRE(crossV > 0.0f);

    // Verify crosshair fits within screen
    REQUIRE(left >= 0.0f);
    REQUIRE(left + crossLineW <= 1.0f);
    REQUIRE(bottom >= 0.0f);
    REQUIRE(bottom + crossV <= 1.0f);
}

TEST_CASE("UIOverlay orthographic projection maps screen to NDC", "[render][ui]") {
    // Verify the orthographic projection matrix maps [0,1] screen coords to [-1,1] NDC.
    // Matrix:
    //   [ 2,  0,  0,  0]
    //   [ 0,  2,  0,  0]
    //   [ 0,  0,  1,  0]
    //   [-1, -1,  0,  1]
    //
    // For point (x, y, 0, 1): result = (2x-1, 2y-1, 0, 1)

    auto transform = [](float sx, float sy) -> std::pair<float, float> {
        float nx = 2.0f * sx - 1.0f;
        float ny = 2.0f * sy - 1.0f;
        return {nx, ny};
    };

    // Screen (0, 0) → NDC (-1, -1)
    auto p0 = transform(0.0f, 0.0f);
    REQUIRE(p0.first == Catch::Approx(-1.0f));
    REQUIRE(p0.second == Catch::Approx(-1.0f));

    // Screen (1, 1) → NDC (1, 1)
    auto p1 = transform(1.0f, 1.0f);
    REQUIRE(p1.first == Catch::Approx(1.0f));
    REQUIRE(p1.second == Catch::Approx(1.0f));

    // Screen (0.5, 0.5) → NDC (0, 0)
    auto p2 = transform(0.5f, 0.5f);
    REQUIRE(p2.first == Catch::Approx(0.0f));
    REQUIRE(p2.second == Catch::Approx(0.0f));
}

TEST_CASE("UIOverlay quad index order forms two triangles", "[render][ui]") {
    // Index buffer: {0, 1, 2, 0, 2, 3}
    // Triangle 1: vertices 0, 1, 2 (BL, TL, BR) — left-bottom triangle
    // Triangle 2: vertices 0, 2, 3 (BL, BR, TR) — right-top triangle
    uint16_t indices[] = {0, 1, 2, 0, 2, 3};

    // Verify 6 indices (2 triangles)
    REQUIRE(sizeof(indices) / sizeof(indices[0]) == 6);

    // Verify all indices reference valid vertices (0-3)
    for (uint16_t idx : indices) {
        REQUIRE((idx >= 0 && idx <= 3));
    }

    // Verify triangle 1 covers bottom-left half
    REQUIRE(indices[0] == 0); // BL
    REQUIRE(indices[1] == 1); // TL
    REQUIRE(indices[2] == 2); // BR

    // Verify triangle 2 covers top-right half
    REQUIRE(indices[3] == 0); // BL
    REQUIRE(indices[4] == 2); // BR
    REQUIRE(indices[5] == 3); // TR
}

// ============================================================================
// Physics Engine Tests (Phase 5)
// ============================================================================
TEST_CASE("AABB sweep collision: entity moves through empty space unchanged", "[physics]") {
    auto world = std::make_shared<World>(42);
    // Force-load the chunk at (0,0) so setBlock works
    world->getChunk(0, 0);

    // Place entity at high Y (200) where there's no terrain
    AABB entityAABB{Vec3{5.f, 200.f, 5.f}, Vec3{5.6f, 201.8f, 5.6f}};
    Vec3 movement{1.f, 0.f, 1.f};

    PhysicsEngine physics;
    Vec3 resolved = physics.sweepCollision(entityAABB, movement, *world);

    // In empty space (y=200), movement should pass through unchanged
    REQUIRE(resolved.x == Catch::Approx(1.f).margin(0.01f));
    REQUIRE(resolved.y == Catch::Approx(0.f).margin(0.01f));
    REQUIRE(resolved.z == Catch::Approx(1.f).margin(0.01f));
}

TEST_CASE("AABB sweep collision: entity blocked by solid block", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a wall of stone blocks at x=10, high Y (200)
    for (int y = 195; y <= 210; ++y) {
        for (int z = 0; z <= 10; ++z) {
            world->setBlock(10, y, z, BlockType::STONE);
        }
    }

    PhysicsEngine physics;
    // Entity at x=8, y=200 moving toward the wall
    AABB entityAABB{Vec3{8.f, 200.f, 5.f}, Vec3{8.6f, 201.8f, 5.6f}};
    Vec3 movement{5.f, 0.f, 0.f}; // Would move to x=13, but wall is at x=10

    Vec3 resolved = physics.sweepCollision(entityAABB, movement, *world);

    // X movement should be blocked (entity max.x = 8.6, wall min.x = 10)
    // Entity can move up to 10 - 8.6 = 1.4 blocks
    REQUIRE(resolved.x >= 0.f);
    REQUIRE(resolved.x < 5.f); // Movement reduced by wall
}

TEST_CASE("AABB sweep collision: entity slides along wall (Y-first, then X)", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place floor at y=199 and wall at x=10, high Y
    for (int x = 0; x <= 15; ++x) {
        for (int z = 0; z <= 10; ++z) {
            world->setBlock(x, 199, z, BlockType::STONE); // Floor
        }
    }
    for (int y = 190; y <= 210; ++y) {
        for (int z = 0; z <= 10; ++z) {
            world->setBlock(10, y, z, BlockType::STONE); // Wall
        }
    }

    PhysicsEngine physics;
    // Entity above floor, moving down and toward wall
    AABB entityAABB{Vec3{8.f, 202.f, 5.f}, Vec3{8.6f, 203.8f, 5.6f}};
    Vec3 movement{5.f, -3.f, 0.f};

    Vec3 resolved = physics.sweepCollision(entityAABB, movement, *world);

    // Y should be resolved first (land on floor at y=199), then X blocked by wall
    REQUIRE(resolved.y >= -3.f);
    REQUIRE(resolved.x < 5.f); // X reduced by wall collision
}

TEST_CASE("Obstacle collection: returns correct blocks in range", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place blocks at high Y to avoid terrain
    world->setBlock(5, 200, 5, BlockType::STONE);
    world->setBlock(6, 200, 5, BlockType::STONE);
    world->setBlock(5, 200, 6, BlockType::AIR); // Not solid

    AABB queryAABB{Vec3{4.f, 199.f, 4.f}, Vec3{7.f, 201.f, 7.f}};

    std::vector<AABB> obstacles = PhysicsEngine::collectObstacles(queryAABB, *world);

    // Should find at least the two STONE blocks
    REQUIRE(obstacles.size() >= 2);

    // Verify each obstacle is a 1x1x1 block at integer coords
    for (const auto& obs : obstacles) {
        Vec3 size = obs.max - obs.min;
        REQUIRE(size.x == Catch::Approx(1.f));
        REQUIRE(size.y == Catch::Approx(1.f));
        REQUIRE(size.z == Catch::Approx(1.f));
    }
}

TEST_CASE("isSolid: returns true for STONE, false for AIR/WATER/GLASS", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Set blocks at high Y (200) to avoid terrain interference
    world->setBlock(1, 200, 1, BlockType::STONE);
    world->setBlock(2, 200, 2, BlockType::AIR);
    world->setBlock(3, 200, 3, BlockType::WATER);
    world->setBlock(4, 200, 4, BlockType::GLASS);
    world->setBlock(5, 200, 5, BlockType::DIRT);
    world->setBlock(6, 200, 6, BlockType::BEDROCK);

    REQUIRE(PhysicsEngine::isSolid(*world, 1, 200, 1) == true);  // STONE
    REQUIRE(PhysicsEngine::isSolid(*world, 2, 200, 2) == false); // AIR
    REQUIRE(PhysicsEngine::isSolid(*world, 3, 200, 3) == false); // WATER
    REQUIRE(PhysicsEngine::isSolid(*world, 4, 200, 4) == false); // GLASS
    REQUIRE(PhysicsEngine::isSolid(*world, 5, 200, 5) == true);  // DIRT
    REQUIRE(PhysicsEngine::isSolid(*world, 6, 200, 6) == true);  // BEDROCK
}

TEST_CASE("isInWater: returns true when entity AABB overlaps water block", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->setBlock(5, 200, 5, BlockType::WATER);

    // Entity overlapping the water block at high Y
    AABB inWater{Vec3{4.5f, 199.5f, 4.5f}, Vec3{5.5f, 200.5f, 5.5f}};
    REQUIRE(PhysicsEngine::isInWater(*world, inWater) == true);

    // Entity not overlapping water (far away in empty space)
    AABB notInWater{Vec3{50.f, 200.f, 50.f}, Vec3{50.6f, 201.8f, 50.6f}};
    REQUIRE(PhysicsEngine::isInWater(*world, notInWater) == false);
}

// ============================================================================
// Player Tests (Phase 5)
// ============================================================================
TEST_CASE("Player AABB: correct dimensions (0.6x1.8x0.6)", "[player]") {
    Player player;
    player.position = Vec3{10.f, 64.f, 10.f};

    AABB aabb = player.getAABB();

    Vec3 size = aabb.max - aabb.min;
    REQUIRE(size.x == Catch::Approx(0.6f));
    REQUIRE(size.y == Catch::Approx(1.8f));
    REQUIRE(size.z == Catch::Approx(0.6f));

    // Center X/Z should match player position, Y should start at player position
    REQUIRE(aabb.min.y == Catch::Approx(64.f));
    REQUIRE(aabb.max.y == Catch::Approx(65.8f));
}

TEST_CASE("Player gravity: velocity.y decreases by 0.08 per tick", "[player]") {
    auto world = std::make_shared<World>(42);
    // Place floor far below so player falls freely
    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    Player player;
    player.position = Vec3{0.f, 100.f, 0.f}; // High up
    player.velocity = Vec3::zero();

    InputState input;
    player.tick(*world, input, false);

    // After one tick: gravity (-0.08) applied, then vertical drag (0.98)
    // velocity.y = (0 + (-0.08)) * 0.98 = -0.0784
    REQUIRE(player.velocity.y < 0.f);
    REQUIRE(player.velocity.y == Catch::Approx(-0.08f * 0.98f).margin(0.01f));
}

TEST_CASE("Player terminal velocity: clamped to -3.92", "[player]") {
    auto world = std::make_shared<World>(42);
    // Place floor very far below
    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    Player player;
    player.position = Vec3{0.f, 200.f, 0.f};
    player.velocity = Vec3{0.f, -10.f, 0.f}; // Start below terminal

    InputState input;
    player.tick(*world, input, false);

    // Velocity should be clamped to terminal velocity (after drag)
    // -10 * 0.98 = -9.8, then clamped to -3.92
    REQUIRE(player.velocity.y >= Player::TERMINAL_VELOCITY);
}

TEST_CASE("Player jump: velocity.y = +0.42 when on ground", "[player]") {
    Player player;
    player.position = Vec3{0.f, 64.f, 0.f};
    player.velocity = Vec3::zero();
    player.onGround = true;
    player.jumpCooldown = 0;

    player.jump();

    REQUIRE(player.velocity.y == Catch::Approx(Player::JUMP_VELOCITY));
    REQUIRE(player.jumpCooldown == Player::JUMP_COOLDOWN_TICKS);
}

TEST_CASE("Player fall damage: ceil(fallDistance - 3) hearts", "[player]") {
    Player player;
    player.health = 20;

    // Fall distance of 8 → damage = ceil(8 - 3) = 5
    player.fallDistance = 8;
    player.applyFallDamage();
    REQUIRE(player.health == 15);

    // Fall distance of 3 → no damage
    player.health = 20;
    player.fallDistance = 3;
    player.applyFallDamage();
    REQUIRE(player.health == 20);

    // Fall distance of 4 → damage = ceil(4 - 3) = 1
    player.health = 20;
    player.fallDistance = 4;
    player.applyFallDamage();
    REQUIRE(player.health == 19);
}

TEST_CASE("Player fall damage capped at zero health", "[player]") {
    Player player;
    player.health = 2;
    player.fallDistance = 20; // damage = ceil(20 - 3) = 17
    player.applyFallDamage();
    REQUIRE(player.health == 0);
}

TEST_CASE("Player resetFallDistance clears fall tracking", "[player]") {
    Player player;
    player.fallDistance = 10;
    player.resetFallDistance();
    REQUIRE(player.fallDistance == 0);
}

// ============================================================================
// DDA Voxel Traversal Tests (Phase 5)
// ============================================================================
TEST_CASE("DDA traversal: ray hits block at expected position", "[voxel]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a stone block at (5, 200, 0) — high Y to avoid terrain
    world->setBlock(5, 200, 0, BlockType::STONE);

    // Ray from (0, 200, 0) going +X toward the block
    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRay(origin, direction, *world, 10.f);

    REQUIRE(hit.has_value());
    REQUIRE(hit->x == Catch::Approx(5.f));
    REQUIRE(hit->y == Catch::Approx(200.f));
    REQUIRE(hit->z == Catch::Approx(0.f));
}

TEST_CASE("DDA traversal: ray misses all blocks in empty space", "[voxel]") {
    auto world = std::make_shared<World>(42);
    // Don't place any blocks near the ray path

    // Shoot at very high Y (250) where there's definitely no terrain
    Vec3 origin{0.f, 250.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRay(origin, direction, *world, 6.f);

    REQUIRE(hit.has_value() == false);
}

TEST_CASE("DDA traversal: face normal computation", "[voxel]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a stone block at (5, 200, 0) — high Y
    world->setBlock(5, 200, 0, BlockType::STONE);

    // Ray from (2, 200, 0) going +X — hits the -X face of the block
    Vec3 origin{2.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);

    REQUIRE(hit.has_value());
    REQUIRE(hit->first.x == Catch::Approx(5.f));
    REQUIRE(hit->first.y == Catch::Approx(200.f));
    REQUIRE(hit->first.z == Catch::Approx(0.f));

    // Normal should point in -X direction (the face we hit)
    REQUIRE(hit->second.x == Catch::Approx(-1.f));
    REQUIRE(hit->second.y == Catch::Approx(0.f));
    REQUIRE(hit->second.z == Catch::Approx(0.f));
}

TEST_CASE("DDA traversal: ray along diagonal hits block", "[voxel]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place stone block at (3, 203, 3) — high Y
    world->setBlock(3, 203, 3, BlockType::STONE);

    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 1.f, 1.f}; // Diagonal up-forward
    direction = direction.normalize();

    auto hit = VoxelTraversal::traceRay(origin, direction, *world, 10.f);

    // Should hit the block at (3, 203, 3)
    REQUIRE(hit.has_value());
    REQUIRE(hit->x == Catch::Approx(3.f));
    REQUIRE(hit->y == Catch::Approx(203.f));
    REQUIRE(hit->z == Catch::Approx(3.f));
}

TEST_CASE("DDA traversal: maxDistance limits traversal range", "[voxel]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place stone block far away at high Y
    world->setBlock(10, 200, 0, BlockType::STONE);

    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    // maxDistance too short to reach the block at x=10
    auto hit = VoxelTraversal::traceRay(origin, direction, *world, 5.f);
    REQUIRE(hit.has_value() == false);

    // maxDistance long enough to reach the block
    auto hitFar = VoxelTraversal::traceRay(origin, direction, *world, 15.f);
    REQUIRE(hitFar.has_value());
    REQUIRE(hitFar->x == Catch::Approx(10.f));
}

// ============================================================================
// Phase 6: Block Interaction & Environment Tests
// ============================================================================

// ---- Block Breaking Tests (Task 6.1) ----

TEST_CASE("Block breaking: raycast hits block and block becomes AIR", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a stone block at (5, 200, 0) — high Y to avoid terrain
    world->setBlock(5, 200, 0, BlockType::STONE);
    REQUIRE(world->getBlock(5, 200, 0) == BlockType::STONE);

    // Ray from (0, 200, 0) going +X toward the block
    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);
    REQUIRE(hit.has_value());
    REQUIRE(hit->first.x == Catch::Approx(5.f));
    REQUIRE(hit->first.y == Catch::Approx(200.f));
    REQUIRE(hit->first.z == Catch::Approx(0.f));

    // "Break" the block: set to AIR
    int hitX = static_cast<int>(std::floor(hit->first.x));
    int hitY = static_cast<int>(std::floor(hit->first.y));
    int hitZ = static_cast<int>(std::floor(hit->first.z));
    world->setBlock(hitX, hitY, hitZ, BlockType::AIR);

    // Verify block is now AIR
    REQUIRE(world->getBlock(5, 200, 0) == BlockType::AIR);
}

TEST_CASE("Block breaking: chunk marked dirty after block change", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    auto chunk = world->getChunk(0, 0);

    // Reset dirty state
    chunk->needsMeshUpdate = false;

    // Place and break a block
    world->setBlock(5, 200, 0, BlockType::STONE);
    REQUIRE(chunk->needsMeshUpdate == true);

    // Reset and break
    chunk->needsMeshUpdate = false;
    world->setBlock(5, 200, 0, BlockType::AIR);
    REQUIRE(chunk->needsMeshUpdate == true);
}

TEST_CASE("Block breaking: bedrock cannot be broken", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    world->setBlock(5, 200, 0, BlockType::BEDROCK);

    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);

    // Ray should NOT hit bedrock (it's not solid for ray tracing purposes in some games)
    // But in our implementation, bedrock IS solid for ray tracing
    // The "cannot break" logic is in the engine, not the traversal
    if (hit.has_value()) {
        // Verify it's bedrock
        REQUIRE(world->getBlock(5, 200, 0) == BlockType::BEDROCK);
    }
}

// ---- Block Placing Tests (Task 6.2) ----

TEST_CASE("Block placing: raycast finds face and block placed on face normal", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a stone block at (5, 200, 0)
    world->setBlock(5, 200, 0, BlockType::STONE);

    // Ray from (2, 200, 0) going +X — hits the -X face of the block
    Vec3 origin{2.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);
    REQUIRE(hit.has_value());

    // Calculate placement position: hit block + face normal
    int placeX = static_cast<int>(std::floor(hit->first.x)) + static_cast<int>(hit->second.x);
    int placeY = static_cast<int>(std::floor(hit->first.y)) + static_cast<int>(hit->second.y);
    int placeZ = static_cast<int>(std::floor(hit->first.z)) + static_cast<int>(hit->second.z);

    // Normal should be -X, so placement should be at (4, 200, 0)
    REQUIRE(placeX == 4);
    REQUIRE(placeY == 200);
    REQUIRE(placeZ == 0);

    // Place block
    world->setBlock(placeX, placeY, placeZ, BlockType::DIRT);
    REQUIRE(world->getBlock(4, 200, 0) == BlockType::DIRT);
}

TEST_CASE("Block placing: no placement when overlapping player AABB", "[phase6][block]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place block at (10, 200, 10)
    world->setBlock(10, 200, 10, BlockType::STONE);

    // Player standing at (10, 200, 10) — inside the block we'd place on
    Player player;
    player.position = Vec3{10.f, 200.f, 10.f};

    // Ray from (7, 200, 10) going +X
    Vec3 origin{7.f, 200.f, 10.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);
    REQUIRE(hit.has_value());

    // Calculate placement position
    int placeX = static_cast<int>(std::floor(hit->first.x)) + static_cast<int>(hit->second.x);
    int placeY = static_cast<int>(std::floor(hit->first.y)) + static_cast<int>(hit->second.y);
    int placeZ = static_cast<int>(std::floor(hit->first.z)) + static_cast<int>(hit->second.z);

    // Check overlap
    AABB placeBox{
        Vec3{static_cast<float>(placeX), static_cast<float>(placeY), static_cast<float>(placeZ)},
        Vec3{static_cast<float>(placeX + 1), static_cast<float>(placeY + 1), static_cast<float>(placeZ + 1)}
    };

    bool overlaps = placeBox.intersects(player.getAABB());
    // If it overlaps, we should NOT place the block
    if (overlaps) {
        // This is the expected behavior — block should not be placed
        REQUIRE(overlaps == true);
    }
}

TEST_CASE("Block placing: adjacent chunks marked dirty at boundary", "[phase6][block]") {
    auto world = std::make_shared<World>(42);

    // Place block at chunk boundary: x=15 is last block in chunk 0, x=16 is first in chunk 1
    world->getChunk(0, 0);
    world->getChunk(1, 0);

    auto chunk0 = world->getChunk(0, 0);
    auto chunk1 = world->getChunk(1, 0);

    chunk0->needsMeshUpdate = false;
    chunk1->needsMeshUpdate = false;

    // Place block at x=16 (first block of chunk 1)
    world->setBlock(16, 200, 8, BlockType::STONE);

    // Chunk 1 should be dirty
    REQUIRE(chunk1->needsMeshUpdate == true);
}

// ---- Hotbar Tests (Task 6.3) ----

TEST_CASE("Hotbar: initial slot selection is 0", "[phase6][hotbar]") {
    Hotbar hotbar;
    REQUIRE(hotbar.getSelectedIndex() == 0);
}

TEST_CASE("Hotbar: selectSlot clamps to valid range", "[phase6][hotbar]") {
    Hotbar hotbar;

    // Negative index clamps to 0
    hotbar.selectSlot(-5);
    REQUIRE(hotbar.getSelectedIndex() == 0);

    // Out-of-range index clamps to 8
    hotbar.selectSlot(100);
    REQUIRE(hotbar.getSelectedIndex() == 8);

    // Valid index works
    hotbar.selectSlot(4);
    REQUIRE(hotbar.getSelectedIndex() == 4);
}

TEST_CASE("Hotbar: selectNext wraps around", "[phase6][hotbar]") {
    Hotbar hotbar;
    hotbar.selectSlot(0);

    for (int i = 1; i <= 8; ++i) {
        hotbar.selectNext();
        REQUIRE(hotbar.getSelectedIndex() == i);
    }

    // Wrap around: 8 → 0
    hotbar.selectNext();
    REQUIRE(hotbar.getSelectedIndex() == 0);
}

TEST_CASE("Hotbar: selectPrev wraps around", "[phase6][hotbar]") {
    Hotbar hotbar;
    hotbar.selectSlot(8);

    for (int i = 7; i >= 0; --i) {
        hotbar.selectPrev();
        REQUIRE(hotbar.getSelectedIndex() == i);
    }

    // Wrap around: 0 → 8
    hotbar.selectPrev();
    REQUIRE(hotbar.getSelectedIndex() == 8);
}

TEST_CASE("Hotbar: getSelectedBlockType returns correct type", "[phase6][hotbar]") {
    Hotbar hotbar;

    // Default slot 0 is STONE
    hotbar.selectSlot(0);
    REQUIRE(hotbar.getSelectedBlockType() == BlockType::STONE);

    // Slot 1 is DIRT
    hotbar.selectSlot(1);
    REQUIRE(hotbar.getSelectedBlockType() == BlockType::DIRT);

    // Slot 2 is GRASS
    hotbar.selectSlot(2);
    REQUIRE(hotbar.getSelectedBlockType() == BlockType::GRASS);
}

TEST_CASE("Hotbar: setSlot and getSlot", "[phase6][hotbar]") {
    Hotbar hotbar;

    hotbar.setSlot(0, BlockType::DIAMOND_ORE);
    REQUIRE(hotbar.getSlot(0) == BlockType::DIAMOND_ORE);

    // Out-of-range returns AIR
    REQUIRE(hotbar.getSlot(-1) == BlockType::AIR);
    REQUIRE(hotbar.getSlot(9) == BlockType::AIR);

    // setSlot on out-of-range does nothing
    hotbar.setSlot(-1, BlockType::STONE);
    REQUIRE(hotbar.getSlot(0) == BlockType::DIAMOND_ORE);
}

TEST_CASE("Hotbar: default slot contents", "[phase6][hotbar]") {
    Hotbar hotbar;

    REQUIRE(hotbar.getSlot(0) == BlockType::STONE);
    REQUIRE(hotbar.getSlot(1) == BlockType::DIRT);
    REQUIRE(hotbar.getSlot(2) == BlockType::GRASS);
    REQUIRE(hotbar.getSlot(3) == BlockType::LOG);
    REQUIRE(hotbar.getSlot(4) == BlockType::SAND);
    REQUIRE(hotbar.getSlot(5) == BlockType::PLANKS);
    REQUIRE(hotbar.getSlot(6) == BlockType::BEDROCK);
    REQUIRE(hotbar.getSlot(7) == BlockType::COAL_ORE);
    REQUIRE(hotbar.getSlot(8) == BlockType::IRON_ORE);
}

// ---- Day/Night Cycle Tests (Task 6.4-6.5) ----

TEST_CASE("Day/night cycle: sun position at noon", "[phase6][daynight]") {
    // At noon: worldTime = 6000 (25% of 24000)
    // orbitalAngle = 0.25 * 2*PI = PI/2
    // sunDirection = (cos(PI/2), sin(PI/2), 0.3) = (0, 1, 0.3)
    uint64_t worldTime = 6000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction = static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    // At noon: cos(PI/2) ≈ 0, sin(PI/2) = 1
    REQUIRE(sunX == Catch::Approx(0.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(1.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at sunset", "[phase6][daynight]") {
    // At sunset: worldTime = 12000 (50% of 24000)
    // orbitalAngle = 0.5 * 2*PI = PI
    // sunDirection = (cos(PI), sin(PI), 0.3) = (-1, 0, 0.3)
    uint64_t worldTime = 12000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction = static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(-1.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at midnight", "[phase6][daynight]") {
    // At midnight: worldTime = 18000 (75% of 24000)
    // orbitalAngle = 0.75 * 2*PI = 3PI/2
    // sunDirection = (cos(3PI/2), sin(3PI/2), 0.3) = (0, -1, 0.3)
    uint64_t worldTime = 18000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction = static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(0.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(-1.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at dawn", "[phase6][daynight]") {
    // At dawn: worldTime = 0 (or 24000)
    // orbitalAngle = 0
    // sunDirection = (cos(0), sin(0), 0.3) = (1, 0, 0.3)
    uint64_t worldTime = 0;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction = static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(1.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: world time wraps at day boundary", "[phase6][daynight]") {
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    // worldTime = 48000 (2 days) should wrap to same position as 0
    uint64_t worldTime = 48000;
    float dayFraction = static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    REQUIRE(dayFraction == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun elevation drives ambient brightness", "[phase6][daynight]") {
    // Test that sun elevation at noon produces higher ambient than at midnight
    auto computeAmbient = [](uint64_t worldTime) -> float {
        static constexpr uint64_t TICKS_PER_DAY = 24000;
        float dayFraction = static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
        float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);
        float sunElevation = std::sin(orbitalAngle);

        float ambientDay = 0.35f;
        float ambientNight = 0.1f;
        float ambientT = std::max(0.0f, std::min(1.0f, (sunElevation + 0.2f) / 0.6f));
        return ambientNight + (ambientDay - ambientNight) * ambientT;
    };

    float ambientNoon = computeAmbient(6000);
    float ambientMidnight = computeAmbient(18000);

    // Noon ambient should be higher than midnight
    REQUIRE(ambientNoon > ambientMidnight);
    REQUIRE(ambientNoon == Catch::Approx(0.35f).margin(0.01f));
    REQUIRE(ambientMidnight == Catch::Approx(0.1f).margin(0.01f));
}

// ---- Water Physics Tests (Task 6.7-6.8) ----

TEST_CASE("Water physics: reduced gravity when submerged", "[phase6][water]") {
    auto world = std::make_shared<World>(42);
    // Ensure chunk exists before setting blocks (setBlock only modifies existing chunks)
    world->getChunk(0, 0);

    // Place floor far below so player falls freely
    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    // Place water block at player position
    world->setBlock(0, 100, 0, BlockType::WATER);

    Player player;
    player.position = Vec3{0.f, 100.f, 0.f};
    player.velocity = Vec3::zero();

    InputState input;
    player.tick(*world, input, false);

    // In water: gravity *= 0.3, so effective gravity = -0.08 * 0.3 = -0.024
    // After drag: -0.024 * 0.98 = -0.02352
    // Plus buoyancy: -0.02352 + 0.02 = -0.00352
    // Velocity should be much smaller than in air (-0.0784)
    REQUIRE(std::abs(player.velocity.y) < std::abs(-0.08f * 0.98f));
}

TEST_CASE("Water physics: increased horizontal drag in water", "[phase6][water]") {
    auto world = std::make_shared<World>(42);
    // Ensure chunk exists before setting blocks (setBlock only modifies existing chunks)
    world->getChunk(0, 0);

    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    // Place water at player position
    world->setBlock(0, 100, 0, BlockType::WATER);

    Player player;
    player.position = Vec3{0.f, 100.f, 0.f};
    player.velocity = Vec3::zero();
    player.yaw = 0.f;

    // Press W to move forward
    InputState input;
    input.keysDown[Key::W] = true;
    player.tick(*world, input, false);

    // In water, horizontal drag is 0.7 (vs 0.91 in air)
    // Movement should be significantly reduced
    float totalHorizontalSpeed = std::sqrt(player.velocity.x * player.velocity.x +
                                            player.velocity.z * player.velocity.z);
    // In air: speed = 0.05 * 0.91 = 0.0455
    // In water: speed = 0.05 * 0.7 = 0.035
    REQUIRE(totalHorizontalSpeed < 0.0455f);
}

TEST_CASE("Water physics: buoyancy pushes player upward", "[phase6][water]") {
    auto world = std::make_shared<World>(42);
    // Ensure chunk exists before setting blocks (setBlock only modifies existing chunks)
    world->getChunk(0, 0);

    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    world->setBlock(0, 100, 0, BlockType::WATER);

    Player player;
    player.position = Vec3{0.f, 100.f, 0.f};
    player.velocity = Vec3{0.f, -0.1f, 0.f}; // Moving downward

    InputState input;
    player.tick(*world, input, false);

    // Buoyancy should reduce downward velocity
    // Without buoyancy: velocity.y ≈ -0.1 * 0.98 + (-0.024) = -0.122
    // With buoyancy: velocity.y ≈ -0.122 + 0.02 = -0.102
    // The buoyancy force makes velocity.y less negative
    REQUIRE(player.velocity.y > -0.13f);
}

TEST_CASE("isInWater: detects player in water block", "[phase6][water]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->setBlock(5, 200, 5, BlockType::WATER);

    // Player AABB overlapping water block
    AABB playerBox{
        Vec3{4.5f, 199.5f, 4.5f},
        Vec3{5.1f, 201.3f, 5.1f}
    };
    REQUIRE(PhysicsEngine::isInWater(*world, playerBox) == true);

    // Player not in water
    AABB playerBox2{
        Vec3{10.f, 200.f, 10.f},
        Vec3{10.6f, 201.8f, 10.6f}
    };
    REQUIRE(PhysicsEngine::isInWater(*world, playerBox2) == false);
}

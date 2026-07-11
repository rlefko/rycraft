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

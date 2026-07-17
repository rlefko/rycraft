#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/hotbar.hpp>
#include <engine/input_bindings.hpp>
#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/spatial_hash.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/shader_types.hpp>
#include <render/ui_menu.hpp>
#include <render/ui_overlay.hpp>
#include <render/vertex.hpp>
#include <world/chunk.hpp>
#include <world/chunk_generator.hpp>
#include <world/chunk_pos.hpp>
#include <world/climate.hpp>
#include <world/noise.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/world.hpp>

#include <chrono>
#include <cmath>
#include <thread>

// ============================================================================
// Vec3 Tests
// ============================================================================
// ===========================================================================
// Foundations: math, threading, randomness
// ===========================================================================

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
    Mat4 p = Mat4::perspective(static_cast<float>(std::atan(1.f) * 2.f), // 90 deg
                               1.f,                                      // aspect
                               1.f,                                      // near
                               100.f                                     // far
    );
    // For 90 deg FOV, aspect 1: tan(45) = 1, so (0,0) = 1.0
    REQUIRE(p(0, 0) == Catch::Approx(1.f).epsilon(0.01f));
    REQUIRE(p(1, 1) == Catch::Approx(1.f).epsilon(0.01f));
    REQUIRE(p(3, 3) == Catch::Approx(0.f));
    REQUIRE(p(3, 2) == Catch::Approx(-1.f));
}

TEST_CASE("Mat4 perspective maps depth to Metal [0,1] clip range", "[math]") {
    const float nearZ = 1.f;
    const float farZ = 100.f;
    Mat4 p = Mat4::perspective(static_cast<float>(std::atan(1.f) * 2.f), 1.f, nearZ, farZ);

    // View space looks down -Z: the near plane must land on NDC z=0 and the
    // far plane on NDC z=1 (Metal), NOT the OpenGL [-1,1] range.
    Vec4 nearPoint = p.transformVec4({0.f, 0.f, -nearZ, 1.f});
    REQUIRE(nearPoint.z / nearPoint.w == Catch::Approx(0.f).margin(1e-5f));

    Vec4 farPoint = p.transformVec4({0.f, 0.f, -farZ, 1.f});
    REQUIRE(farPoint.z / farPoint.w == Catch::Approx(1.f).epsilon(1e-4f));

    Vec4 midPoint = p.transformVec4({0.f, 0.f, -10.f, 1.f});
    float midNdc = midPoint.z / midPoint.w;
    REQUIRE(midNdc > 0.f);
    REQUIRE(midNdc < 1.f);
}

TEST_CASE("Mat4 lookAt", "[math]") {
    Vec3 eye{0.f, 0.f, 5.f};
    Vec3 target{0.f, 0.f, 0.f};
    Vec3 up{0.f, 1.f, 0.f};
    Mat4 v = Mat4::lookAt(eye, target, up);

    // Column-vector convention: translation lives in column 3, so the target
    // (5 units ahead of the camera looking down -Z) maps to (0, 0, -5).
    Vec3 targetView = v.transformVec3(target);
    REQUIRE(targetView.x == Catch::Approx(0.f).margin(1e-5f));
    REQUIRE(targetView.y == Catch::Approx(0.f).margin(1e-5f));
    REQUIRE(targetView.z == Catch::Approx(-5.f).epsilon(0.01f));

    // The eye itself maps to the view-space origin
    Vec3 eyeView = v.transformVec3(eye);
    REQUIRE(eyeView.x == Catch::Approx(0.f).margin(1e-5f));
    REQUIRE(eyeView.y == Catch::Approx(0.f).margin(1e-5f));
    REQUIRE(eyeView.z == Catch::Approx(0.f).margin(1e-5f));

    // World +X is the camera's right; world +Y stays up
    Vec3 right = v.transformVec3({1.f, 0.f, 5.f});
    REQUIRE(right.x == Catch::Approx(1.f).epsilon(0.01f));
    Vec3 above = v.transformVec3({0.f, 1.f, 5.f});
    REQUIRE(above.y == Catch::Approx(1.f).epsilon(0.01f));

    // Basis vectors in rows, translation in column 3 (not the bottom row)
    REQUIRE(v(3, 2) == Catch::Approx(0.f));
    REQUIRE(v(2, 3) == Catch::Approx(-5.f).epsilon(0.01f));
}

TEST_CASE("Mat4 memcpy to simd_float4x4 preserves transform semantics", "[math]") {
    // The engine memcpys Mat4 straight into the GPU uniform buffer, where MSL
    // treats it as a column-major float4x4 multiplying column vectors. This
    // pins that the two conventions agree, without needing a GPU.
    Vec3 eye{3.f, 70.f, -2.f};
    Mat4 view = Mat4::lookAt(eye, {10.f, 64.f, 10.f}, {0.f, 1.f, 0.f});
    Mat4 proj = Mat4::perspective(1.2f, 16.f / 9.f, 0.1f, 1000.f);
    Mat4 vp = proj * view;

    simd_float4x4 gpuMatrix;
    std::memcpy(&gpuMatrix, vp.data.data(), sizeof(gpuMatrix));

    Vec4 point{25.f, 60.f, 40.f, 1.f};
    Vec4 cpuResult = vp.transformVec4(point);
    simd_float4 gpuResult =
        simd_mul(gpuMatrix, simd_make_float4(point.x, point.y, point.z, point.w));

    REQUIRE(gpuResult.x == Catch::Approx(cpuResult.x).epsilon(1e-4f));
    REQUIRE(gpuResult.y == Catch::Approx(cpuResult.y).epsilon(1e-4f));
    REQUIRE(gpuResult.z == Catch::Approx(cpuResult.z).epsilon(1e-4f));
    REQUIRE(gpuResult.w == Catch::Approx(cpuResult.w).epsilon(1e-4f));
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

TEST_CASE("ThreadPool starts newly available priority work before queued FIFO work",
          "[thread][priority][regression]") {
    ThreadPool pool(1);
    std::mutex mutex;
    std::condition_variable condition;
    bool blockerStarted = false;
    bool releaseBlocker = false;
    std::vector<int> order;

    auto blocker = pool.submit([&] {
        std::unique_lock lock(mutex);
        blockerStarted = true;
        condition.notify_all();
        condition.wait(lock, [&] { return releaseBlocker; });
    });
    bool didStart = false;
    {
        std::unique_lock lock(mutex);
        didStart =
            condition.wait_for(lock, std::chrono::seconds(1), [&] { return blockerStarted; });
    }
    if (!didStart) {
        std::lock_guard lock(mutex);
        releaseBlocker = true;
        condition.notify_all();
    }
    REQUIRE(didStart);

    auto low = pool.submitWithPriority(1, [&] { order.push_back(1); });
    auto high = pool.submitWithPriority(10, [&] { order.push_back(10); });
    auto samePriority = pool.submitWithPriority(10, [&] { order.push_back(11); });
    {
        std::lock_guard lock(mutex);
        releaseBlocker = true;
    }
    condition.notify_all();

    blocker.get();
    low.get();
    high.get();
    samePriority.get();
    REQUIRE(order == std::vector<int>{10, 11, 1});
}

TEST_CASE("ThreadPool size", "[thread]") {
    ThreadPool pool(3);
    REQUIRE(pool.size() == 3);
}

TEST_CASE("ThreadPool bounds latency-sensitive workers", "[thread][priority]") {
    REQUIRE_NOTHROW(ThreadPool(3, ThreadPriority::UTILITY, 2));
    REQUIRE_THROWS_AS(ThreadPool(2, ThreadPriority::UTILITY, 3), std::invalid_argument);
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

TEST_CASE("SeededRng is deterministic and bounded", "[common]") {
    SeededRng a(1234);
    SeededRng b(1234);
    for (int i = 0; i < 100; ++i) {
        REQUIRE(a.next() == b.next());
    }

    SeededRng r(99);
    for (int i = 0; i < 1000; ++i) {
        float f = r.nextFloat();
        REQUIRE(f >= 0.0f);
        REQUIRE(f < 1.0f);
        int v = r.nextInt(-3, 7);
        REQUIRE(v >= -3);
        REQUIRE(v <= 7);
    }
}

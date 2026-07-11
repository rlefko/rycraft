#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// Vec2 — 2D SIMD-friendly vector (alignas(16) for SSE/NEON)
// ---------------------------------------------------------------------------
struct alignas(16) Vec2 {
  float x = 0.f;
  float y = 0.f;

  constexpr Vec2() = default;
  constexpr Vec2(float x, float y) : x(x), y(y) {}

  // Arithmetic
  constexpr Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
  constexpr Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
  constexpr Vec2 operator*(float s) const { return {x * s, y * s}; }
  constexpr Vec2 operator/(float s) const {
    float r = 1.f / s;
    return {x * r, y * r};
  }

  // Compound assignment
  constexpr Vec2& operator+=(const Vec2& o) {
    x += o.x;
    y += o.y;
    return *this;
  }
  constexpr Vec2& operator-=(const Vec2& o) {
    x -= o.x;
    y -= o.y;
    return *this;
  }
  constexpr Vec2& operator*=(float s) {
    x *= s;
    y *= s;
    return *this;
  }
  constexpr Vec2& operator/=(float s) {
    float r = 1.f / s;
    x *= r;
    y *= r;
    return *this;
  }

  // Comparison
  constexpr bool operator==(const Vec2& o) const { return x == o.x && y == o.y; }
  constexpr bool operator!=(const Vec2& o) const { return !(*this == o); }

  // Scalar ops
  constexpr float dot(const Vec2& o) const { return x * o.x + y * o.y; }
  float lengthSq() const { return x * x + y * y; }
  float length() const { return std::sqrt(lengthSq()); }
  Vec2 normalize() const {
    float len = length();
    return len > 0.f ? *this / len : Vec2{0.f, 0.f};
  }

  // Interpolation
  constexpr Vec2 lerp(const Vec2& target, float t) const {
    return {x + (target.x - x) * t, y + (target.y - y) * t};
  }

  static constexpr Vec2 zero() { return {0.f, 0.f}; }
  static constexpr Vec2 one() { return {1.f, 1.f}; }
};

// Free scalar * vector
constexpr Vec2 operator*(float s, const Vec2& v) { return v * s; }

// ---------------------------------------------------------------------------
// Vec3 — 3D SIMD-friendly vector (alignas(16) for SSE/NEON)
// ---------------------------------------------------------------------------
struct alignas(16) Vec3 {
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;

  constexpr Vec3() = default;
  constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

  // Arithmetic
  constexpr Vec3 operator+(const Vec3& o) const {
    return {x + o.x, y + o.y, z + o.z};
  }
  constexpr Vec3 operator-(const Vec3& o) const {
    return {x - o.x, y - o.y, z - o.z};
  }
  constexpr Vec3 operator*(float s) const {
    return {x * s, y * s, z * s};
  }
  constexpr Vec3 operator/(float s) const {
    float r = 1.f / s;
    return {x * r, y * r, z * r};
  }

  // Compound assignment
  constexpr Vec3& operator+=(const Vec3& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  constexpr Vec3& operator-=(const Vec3& o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
  }
  constexpr Vec3& operator*=(float s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
  constexpr Vec3& operator/=(float s) {
    float r = 1.f / s;
    x *= r;
    y *= r;
    z *= r;
    return *this;
  }

  // Unary negation
  constexpr Vec3 operator-() const { return {-x, -y, -z}; }

  // Comparison
  constexpr bool operator==(const Vec3& o) const {
    return x == o.x && y == o.y && z == o.z;
  }
  constexpr bool operator!=(const Vec3& o) const { return !(*this == o); }

  // Scalar ops
  constexpr float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
  constexpr Vec3 cross(const Vec3& o) const {
    return {
        y * o.z - z * o.y,
        z * o.x - x * o.z,
        x * o.y - y * o.x,
    };
  }
  constexpr float lengthSq() const { return x * x + y * y + z * z; }
  float length() const { return std::sqrt(lengthSq()); }
  Vec3 normalize() const {
    float len = length();
    return len > 0.f ? *this / len : Vec3{0.f, 0.f, 0.f};
  }

  // Interpolation
  constexpr Vec3 lerp(const Vec3& target, float t) const {
    return {x + (target.x - x) * t,
            y + (target.y - y) * t,
            z + (target.z - z) * t};
  }

  static constexpr Vec3 zero() { return {0.f, 0.f, 0.f}; }
  static constexpr Vec3 one() { return {1.f, 1.f, 1.f}; }
  static constexpr Vec3 up() { return {0.f, 1.f, 0.f}; }
  static constexpr Vec3 forward() { return {0.f, 0.f, -1.f}; }
};

// Free scalar * vector
constexpr Vec3 operator*(float s, const Vec3& v) { return v * s; }

// ---------------------------------------------------------------------------
// Vec4 — 4D SIMD-friendly vector (alignas(16) for SSE/NEON)
// ---------------------------------------------------------------------------
struct alignas(16) Vec4 {
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
  float w = 0.f;

  constexpr Vec4() = default;
  constexpr Vec4(float x, float y, float z, float w)
      : x(x), y(y), z(z), w(w) {}

  // Construct from Vec3 + w
  constexpr Vec4(const Vec3& v, float w)
      : x(v.x), y(v.y), z(v.z), w(w) {}

  // Arithmetic
  constexpr Vec4 operator+(const Vec4& o) const {
    return {x + o.x, y + o.y, z + o.z, w + o.w};
  }
  constexpr Vec4 operator-(const Vec4& o) const {
    return {x - o.x, y - o.y, z - o.z, w - o.w};
  }
  constexpr Vec4 operator*(float s) const {
    return {x * s, y * s, z * s, w * s};
  }
  constexpr Vec4 operator/(float s) const {
    float r = 1.f / s;
    return {x * r, y * r, z * r, w * r};
  }

  // Compound assignment
  constexpr Vec4& operator+=(const Vec4& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    w += o.w;
    return *this;
  }
  constexpr Vec4& operator-=(const Vec4& o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    w -= o.w;
    return *this;
  }
  constexpr Vec4& operator*=(float s) {
    x *= s;
    y *= s;
    z *= s;
    w *= s;
    return *this;
  }
  constexpr Vec4& operator/=(float s) {
    float r = 1.f / s;
    x *= r;
    y *= r;
    z *= r;
    w *= r;
    return *this;
  }

  // Comparison
  constexpr bool operator==(const Vec4& o) const {
    return x == o.x && y == o.y && z == o.z && w == o.w;
  }
  constexpr bool operator!=(const Vec4& o) const { return !(*this == o); }

  static constexpr Vec4 zero() { return {0.f, 0.f, 0.f, 0.f}; }
  static constexpr Vec4 one() { return {1.f, 1.f, 1.f, 1.f}; }
};

// Free scalar * vector
constexpr Vec4 operator*(float s, const Vec4& v) { return v * s; }

// ---------------------------------------------------------------------------
// Mat4 — Column-major 4×4 matrix (16 floats, alignas(16))
// ---------------------------------------------------------------------------
struct alignas(16) Mat4 {
  std::array<float, 16> data;

  constexpr Mat4() : data{} { data.fill(0.f); }

  // Index: column-major  m[col * 4 + row]
  constexpr float& operator()(size_t row, size_t col) {
    return data[col * 4 + row];
  }
  constexpr float operator()(size_t row, size_t col) const {
    return data[col * 4 + row];
  }

  // Comparison
  constexpr bool operator==(const Mat4& o) const { return data == o.data; }
  constexpr bool operator!=(const Mat4& o) const { return !(*this == o); }

  // Matrix multiplication
  constexpr Mat4 operator*(const Mat4& o) const {
    Mat4 result;
    for (size_t c = 0; c < 4; ++c) {
      for (size_t r = 0; r < 4; ++r) {
        float sum = 0.f;
        for (size_t k = 0; k < 4; ++k) {
          sum += (*this)(r, k) * o(k, c);
        }
        result(r, c) = sum;
      }
    }
    return result;
  }

  // Transform Vec4 (column vector)
  constexpr Vec4 transformVec4(const Vec4& v) const {
    return {
        (*this)(0, 0) * v.x + (*this)(0, 1) * v.y + (*this)(0, 2) * v.z +
            (*this)(0, 3) * v.w,
        (*this)(1, 0) * v.x + (*this)(1, 1) * v.y + (*this)(1, 2) * v.z +
            (*this)(1, 3) * v.w,
        (*this)(2, 0) * v.x + (*this)(2, 1) * v.y + (*this)(2, 2) * v.z +
            (*this)(2, 3) * v.w,
        (*this)(3, 0) * v.x + (*this)(3, 1) * v.y + (*this)(3, 2) * v.z +
            (*this)(3, 3) * v.w,
    };
  }

  // Transform Vec3 (homogeneous, w=1)
  constexpr Vec3 transformVec3(const Vec3& v) const {
    Vec4 h = transformVec4({v.x, v.y, v.z, 1.f});
    return {h.x / h.w, h.y / h.w, h.z / h.w};
  }

  // ---- Static factory methods ----

  static constexpr Mat4 identity() {
    Mat4 m;
    m(0, 0) = 1.f;
    m(1, 1) = 1.f;
    m(2, 2) = 1.f;
    m(3, 3) = 1.f;
    return m;
  }

  static Mat4 perspective(float fovY, float aspect, float near, float far) {
    Mat4 m;
    float tanHalfFov = std::tan(fovY * 0.5f);
    m(0, 0) = 1.f / (aspect * tanHalfFov);
    m(1, 1) = 1.f / tanHalfFov;
    m(2, 2) = -(far + near) / (far - near);
    m(2, 3) = -2.f * far * near / (far - near);
    m(3, 2) = -1.f;
    m(3, 3) = 0.f;
    return m;
  }

  static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    Vec3 z = (eye - target).normalize();
    Vec3 x = up.cross(z).normalize();
    Vec3 y = z.cross(x);
    Mat4 m = Mat4::identity();
    m(0, 0) = x.x;
    m(1, 0) = x.y;
    m(2, 0) = x.z;
    m(0, 1) = y.x;
    m(1, 1) = y.y;
    m(2, 1) = y.z;
    m(0, 2) = z.x;
    m(1, 2) = z.y;
    m(2, 2) = z.z;
    m(3, 0) = -x.dot(eye);
    m(3, 1) = -y.dot(eye);
    m(3, 2) = -z.dot(eye);
    return m;
  }

  static constexpr Mat4 translation(float x, float y, float z) {
    Mat4 m = Mat4::identity();
    m(0, 3) = x;
    m(1, 3) = y;
    m(2, 3) = z;
    return m;
  }

  static constexpr Mat4 translation(const Vec3& v) {
    return translation(v.x, v.y, v.z);
  }

  static constexpr Mat4 rotationX(float angle) {
    Mat4 m = Mat4::identity();
    float c = std::cos(angle);
    float s = std::sin(angle);
    m(1, 1) = c;
    m(1, 2) = -s;
    m(2, 1) = s;
    m(2, 2) = c;
    return m;
  }

  static constexpr Mat4 rotationY(float angle) {
    Mat4 m = Mat4::identity();
    float c = std::cos(angle);
    float s = std::sin(angle);
    m(0, 0) = c;
    m(0, 2) = s;
    m(2, 0) = -s;
    m(2, 2) = c;
    return m;
  }

  static constexpr Mat4 rotationZ(float angle) {
    Mat4 m = Mat4::identity();
    float c = std::cos(angle);
    float s = std::sin(angle);
    m(0, 0) = c;
    m(0, 1) = -s;
    m(1, 0) = s;
    m(1, 1) = c;
    return m;
  }

  static constexpr Mat4 scale(float x, float y, float z) {
    Mat4 m;
    m(0, 0) = x;
    m(1, 1) = y;
    m(2, 2) = z;
    m(3, 3) = 1.f;
    return m;
  }

  static constexpr Mat4 scale(const Vec3& v) {
    return scale(v.x, v.y, v.z);
  }

  static constexpr Mat4 scale(float s) {
    return scale(s, s, s);
  }
};

// ---------------------------------------------------------------------------
// AABB — Axis-Aligned Bounding Box
// ---------------------------------------------------------------------------
struct AABB {
  Vec3 min;
  Vec3 max;

  constexpr AABB() : min{0.f, 0.f, 0.f}, max{0.f, 0.f, 0.f} {}
  constexpr AABB(const Vec3& min, const Vec3& max) : min(min), max(max) {}

  // Point containment
  constexpr bool contains(const Vec3& p) const {
    return p.x >= min.x && p.x <= max.x &&
           p.y >= min.y && p.y <= max.y &&
           p.z >= min.z && p.z <= max.z;
  }

  // AABB-AABB intersection
  constexpr bool intersects(const AABB& other) const {
    return min.x <= other.max.x && max.x >= other.min.x &&
           min.y <= other.max.y && max.y >= other.min.y &&
           min.z <= other.max.z && max.z >= other.min.z;
  }

  // Expand by a vector (offset both min and max)
  constexpr AABB expandedBy(const Vec3& v) const {
    return {
        Vec3{min.x + v.x, min.y + v.y, min.z + v.z},
        Vec3{max.x + v.x, max.y + v.y, max.z + v.z},
    };
  }

  // Ray-AABB intersection (slab method)
  bool intersectsRay(const Vec3& origin, const Vec3& dir) const {
    float tMin = -std::numeric_limits<float>::infinity();
    float tMax = std::numeric_limits<float>::infinity();

    float invDir[3] = {1.f / dir.x, 1.f / dir.y, 1.f / dir.z};
    float boxMin[3] = {this->min.x, this->min.y, this->min.z};
    float boxMax[3] = {this->max.x, this->max.y, this->max.z};
    float originC[3] = {origin.x, origin.y, origin.z};

    for (int axis = 0; axis < 3; ++axis) {
      float t1 = (boxMin[axis] - originC[axis]) * invDir[axis];
      float t2 = (boxMax[axis] - originC[axis]) * invDir[axis];

      if (invDir[axis] < 0.f) std::swap(t1, t2);

      tMin = std::max(tMin, t1);
      tMax = std::min(tMax, t2);

      if (tMin > tMax) return false;
    }

    return tMax >= 0.f;
  }

  // Volume
  constexpr float volume() const {
    Vec3 dims = max - min;
    return dims.x * dims.y * dims.z;
  }

  // Center
  constexpr Vec3 center() const {
    return (min + max) * 0.5f;
  }

  // Size
  constexpr Vec3 size() const { return max - min; }
};

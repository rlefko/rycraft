#pragma once

struct Vec3 {
  float x, y, z;

  static constexpr Vec3 zero() {
    return {0.f, 0.f, 0.f};
  }
};

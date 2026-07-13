#pragma once

#include <common/math.hpp>
#include <engine/input.hpp>

#include <cmath>

// Forward declaration — defined in input_bindings.hpp
struct InputBindings;

// ---------------------------------------------------------------------------
// Camera — First-person camera with yaw/pitch, head bobbing, and FOV
//
// Yaw:   mouse X delta → rotate around world Y axis
// Pitch: mouse Y delta → rotate around local X axis, clamped ±89°
// ---------------------------------------------------------------------------
class Camera {
public:
    Camera();

    // Update orientation from mouse delta and movement from keys
    void update(double deltaTime, const InputState& input, const InputBindings& bindings,
                const Vec3& playerPos);

    Mat4 viewMatrix() const;
    Vec3 position() const;
    Vec3 forward() const;
    float yaw() const;
    Vec3 right() const;
    Vec3 up() const;

    void setPosition(const Vec3& pos);
    const Vec3& getPosition() const;

    // Head bobbing — sine wave offset proportional to horizontal speed
    Vec3 getBobOffset() const;

    void setFOV(float fov);
    float FOV() const;

    void setMouseSensitivity(float sensitivity);
    float mouseSensitivity() const;

private:
    Vec3 position_;
    float pitch_ = 0; // radians, clamped to [-89°, 89°]
    float yaw_ = 0;   // radians, wrapped to [-PI, PI]

    float moveSpeed_ = 5.0f;
    float mouseSensitivity_ = 0.002f;
    float fov_ = 70.0f;

    // Head bobbing state
    float bobPhase_ = 0;
    static constexpr float BOB_AMPLITUDE = 0.05f;
    static constexpr float BOB_FREQUENCY = 8.0f;

    // Previous position for speed-based bobbing
    Vec3 previousPosition_;

    Vec3 front_;
    void updateFront();
};

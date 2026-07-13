#include <engine/camera.hpp>

#include <common/math.hpp>
#include <engine/input_bindings.hpp>

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

Camera::Camera()
    : position_{0.f, 2.f, 0.f}
    , pitch_(0)
    , yaw_(0)
    , moveSpeed_(5.0f)
    , mouseSensitivity_(0.002f)
    , fov_(70.0f)
    , bobPhase_(0)
    , previousPosition_{position_}
    , front_{0.f, 0.f, -1.f} {}

void Camera::update(double deltaTime, const InputState& input, const InputBindings& bindings,
                    const Vec3& playerPos) {
    // Store current position for speed calculation
    previousPosition_ = position_;

    // Update position from player
    setPosition(playerPos);

    // ---- Mouse look ----
    Vec2 delta = input.mouseDelta;

    // Yaw from X delta (inverted for natural FPS feel)
    yaw_ -= delta.x * mouseSensitivity_;

    // Pitch from Y delta
    pitch_ += delta.y * mouseSensitivity_;

    // Clamp pitch to [-89°, 89°] to prevent gimbal lock
    float maxPitch = 89.0f * (float(M_PI) / 180.0f);
    pitch_ = std::clamp(pitch_, -maxPitch, maxPitch);

    // Wrap yaw to [-PI, PI]
    while (yaw_ > float(M_PI))
        yaw_ -= 2.f * float(M_PI);
    while (yaw_ < -float(M_PI))
        yaw_ += 2.f * float(M_PI);

    // Recalculate forward vector
    updateFront();

    // ---- Head bobbing ----
    // Calculate horizontal movement speed
    Vec3 horizontalDelta = position_ - previousPosition_;
    horizontalDelta.y = 0.f; // ignore vertical movement
    float dt = deltaTime > 0 ? deltaTime : 1.0 / 60.0;
    float horizontalSpeed = horizontalDelta.length() / dt;

    // Only bob when moving horizontally
    if (horizontalSpeed > 0.01f) {
        // Scale bob frequency by movement speed relative to walk speed
        float speedRatio = std::min(horizontalSpeed / moveSpeed_, 2.0f);
        bobPhase_ += BOB_FREQUENCY * speedRatio * deltaTime * float(M_PI) * 2.0f;
    } else {
        // Smoothly decay bob phase when not moving
        bobPhase_ *= 0.9f;
    }

    // Suppress unused parameter warning — bindings used for key lookup in Phase 3
    (void)bindings;
}

Mat4 Camera::viewMatrix() const {
    Vec3 camUp = Vec3::up();
    return Mat4::lookAt(position_, position_ + front_, camUp);
}

Vec3 Camera::position() const {
    return position_;
}

Vec3 Camera::forward() const {
    return front_;
}

float Camera::yaw() const {
    return yaw_;
}

Vec3 Camera::right() const {
    // Right vector: cross(forward, worldUp)
    return front_.cross(Vec3::up()).normalize();
}

Vec3 Camera::up() const {
    // Camera's local up: cross(right, forward)
    Vec3 r = right();
    return r.cross(front_).normalize();
}

void Camera::setPosition(const Vec3& pos) {
    position_ = pos;
}

const Vec3& Camera::getPosition() const {
    return position_;
}

Vec3 Camera::getBobOffset() const {
    float bobY = std::sin(bobPhase_) * BOB_AMPLITUDE;
    // Subtle horizontal bob (half frequency, quarter amplitude)
    float bobX = std::sin(bobPhase_ * 0.5f) * (BOB_AMPLITUDE * 0.25f);
    return {bobX, bobY, 0.f};
}

void Camera::setFOV(float fov) {
    fov_ = std::clamp(fov, 30.0f, 120.0f);
}

float Camera::FOV() const {
    return fov_;
}

void Camera::setMouseSensitivity(float sensitivity) {
    mouseSensitivity_ = std::max(0.0001f, sensitivity);
}

float Camera::mouseSensitivity() const {
    return mouseSensitivity_;
}

void Camera::updateFront() {
    // Forward vector from spherical coordinates
    // yaw rotates around Y axis, pitch rotates around local X axis
    // cos^2(pitch)*(sin^2(yaw)+cos^2(yaw)) + sin^2(pitch) = 1,
    // so the result is already a unit vector — no normalize needed.
    front_.x = std::cos(pitch_) * std::sin(yaw_);
    front_.y = std::sin(pitch_);
    front_.z = std::cos(pitch_) * std::cos(yaw_);
}

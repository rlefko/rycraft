#pragma once

#include <common/math.hpp>
#include <common/result.hpp>
#include <common/error.hpp>

#include <string>
#include <unordered_map>

#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#endif

// ---------------------------------------------------------------------------
// Key — Scancode-based key identifiers
//
// Ordered by frequency of use for cache-friendly layout.
// ---------------------------------------------------------------------------
enum class Key {
    None,
    Space,
    LeftShift,
    LeftControl,
    W,
    A,
    S,
    D,
    Q,
    E,
    F,
    R,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    Escape,
    Tab,
    Up,
    Down,
    Left,
    Right,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    // Mouse
    MouseLeft,
    MouseRight,
    MouseMiddle
};

// Convert Key enum to/from human-readable strings (for bindings I/O)
std::string keyToString(Key key);
Key keyFromString(const std::string& str);

// ---------------------------------------------------------------------------
// InputState — Frame-level snapshot of all input
//
// Just-pressed / just-released are cleared each frame by update().
// Mouse delta is consumed by the camera each frame.
// ---------------------------------------------------------------------------
struct InputState {
    std::unordered_map<Key, bool> keysDown;
    std::unordered_map<Key, bool> keysJustPressed;
    std::unordered_map<Key, bool> keysJustReleased;
    Vec2 mouseDelta;
    Vec2 mousePosition;
    bool mouseLeftDown = false;
    bool mouseRightDown = false;

    bool isDown(Key key) const;
    bool isJustPressed(Key key) const;
    bool isJustReleased(Key key) const;

    void update();
    void clearMouseDelta();
};

// ---------------------------------------------------------------------------
// InputManager — Cocoa event interception via NSResponder
//
// Uses addLocalMonitorForEventsMatchingMask:handler: to capture
// keyboard and mouse events before they reach the responder chain.
// ---------------------------------------------------------------------------
#ifdef __OBJC__
class InputManager {
public:
    explicit InputManager(NSWindow* window);
    ~InputManager();

    InputManager(const InputManager&) = delete;
    InputManager& operator=(const InputManager&) = delete;

    InputState& state();

    // NSEvent handlers (called from event monitor blocks)
    void handleKeyDown(NSEvent* event);
    void handleKeyUp(NSEvent* event);
    void handleMouseMoved(NSEvent* event);
    void handleMouseDragged(NSEvent* event);
    void handleMouseDown(NSEvent* event);
    void handleMouseUp(NSEvent* event);

    // Cursor control
    void hideAndConfineCursor();
    void showCursor();

private:
    InputState state_;
    NSWindow* window_ = nil;

    // Cursor tracking
    Vec2 lastMousePosition_;
    bool cursorActive_ = false;

    // Event monitors
    id keyDownMonitor_ = nil;
    id keyUpMonitor_ = nil;
    id mouseMovedMonitor_ = nil;
    id mouseDraggedMonitor_ = nil;
    id mouseDownMonitor_ = nil;
    id mouseUpMonitor_ = nil;

    static Key keyCodeToKey(NSInteger keyCode);
};
#endif  // __OBJC__

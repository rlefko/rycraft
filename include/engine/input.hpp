#pragma once

#include <common/error.hpp>
#include <common/math.hpp>

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
    // Presses accumulated for the 20 Hz game tick. keysJustPressed clears
    // every FRAME (60 fps), but most frames run zero ticks — without this,
    // jumps, hotbar keys, and block clicks were silently dropped whenever
    // the press landed on a tickless frame.
    std::unordered_map<Key, bool> keysPressedForTick;
    // Double-tap gestures (sprint on W, fly toggle on Space). Latched at
    // press time and cleared by clearTickPresses(), like keysPressedForTick,
    // so a tap pair isn't dropped on tickless frames. Timing uses the
    // monotonic event clock, not ticks: a frame hitch can run several ticks
    // in a burst, which would misread the gesture's real-time spacing.
    std::unordered_map<Key, double> lastPressTimes; // monotonic seconds
    std::unordered_map<Key, bool> keysDoubleTappedForTick;
    static constexpr double DOUBLE_TAP_WINDOW = 0.3; // seconds between presses
    Vec2 mouseDelta;                                 // accumulated raw look deltas while captured
    Vec2 mousePosition;                              // window points, bottom-left origin
    float scrollDelta = 0.f;                         // accumulated scroll-wheel Y this frame
    bool mouseLeftDown = false;
    bool mouseRightDown = false;

    // Text entry (world name and seed fields). While active, printable
    // ASCII accumulates in textBuffer and game keys are suppressed except
    // Escape, so typed WASD never leaks into movement.
    bool textEntryActive = false;
    std::string textBuffer;
    bool textSubmitted = false; // Return pressed this frame; cleared by update()
    static constexpr size_t TEXT_BUFFER_MAX = 64;

    void beginTextEntry(const std::string& initial);
    std::string endTextEntry(); // deactivates and returns the buffer
    // Pure editing rules so gesture-free tests need no Cocoa events.
    void applyTextKey(char c); // printable 0x20..0x7E, capped at TEXT_BUFFER_MAX
    void applyTextBackspace();

    bool isDown(Key key) const;
    bool isJustPressed(Key key) const;
    bool isJustReleased(Key key) const;

    // Record a physical key press (down + both edge maps) and detect a
    // double-tap against the previous press time. Pure C++ so gesture
    // timing stays unit-testable without Cocoa events.
    void recordPress(Key key, double timeSeconds);

    // Edge-since-last-tick (consumed by clearTickPresses at tick end)
    bool isPressedForTick(Key key) const;
    bool isDoubleTappedForTick(Key key) const;
    void clearTickPresses();

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

    void handleScrollWheel(NSEvent* event);
    void handleFlagsChanged(NSEvent* event);

    // Pointer lock. While captured the hardware cursor is hidden and frozen
    // (CGAssociateMouseAndMouseCursorPosition) and look input comes from raw
    // NSEvent deltas, so the cursor can never wander out of the window
    // mid-play. Release restores a normal cursor centered in the window.
    void captureMouse();
    void releaseMouse();
    bool isMouseCaptured() const { return captured_; }

private:
    InputState state_;
    NSWindow* window_ = nil;

    // Cursor state
    Vec2 lastMousePosition_;
    bool captured_ = false;
    bool cursorHidden_ = false; // NSCursor hide/unhide must stay balanced

    // Move the hardware cursor to the window center (CG coordinates).
    void warpCursorToWindowCenter();

    // Event monitors
    id keyDownMonitor_ = nil;
    id keyUpMonitor_ = nil;
    id mouseMovedMonitor_ = nil;
    id mouseDraggedMonitor_ = nil;
    id mouseDownMonitor_ = nil;
    id mouseUpMonitor_ = nil;
    id scrollWheelMonitor_ = nil;
    id flagsChangedMonitor_ = nil;

    static Key keyCodeToKey(NSInteger keyCode);
};
#endif // __OBJC__

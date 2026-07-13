#include <engine/input.hpp>

#include <common/error.hpp>
#include <common/math.hpp>

#include <cassert>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Key ↔ String conversion
// ---------------------------------------------------------------------------
std::string keyToString(Key key) {
    switch (key) {
        case Key::None:        return "None";
        case Key::Space:       return "Space";
        case Key::LeftShift:   return "LeftShift";
        case Key::LeftControl: return "LeftControl";
        case Key::W:           return "W";
        case Key::A:           return "A";
        case Key::S:           return "S";
        case Key::D:           return "D";
        case Key::Q:           return "Q";
        case Key::E:           return "E";
        case Key::F:           return "F";
        case Key::R:           return "R";
        case Key::F1:          return "F1";
        case Key::F2:          return "F2";
        case Key::F3:          return "F3";
        case Key::F4:          return "F4";
        case Key::F5:          return "F5";
        case Key::F6:          return "F6";
        case Key::F7:          return "F7";
        case Key::F8:          return "F8";
        case Key::F9:          return "F9";
        case Key::F10:         return "F10";
        case Key::Escape:      return "Escape";
        case Key::Tab:         return "Tab";
        case Key::Up:          return "Up";
        case Key::Down:        return "Down";
        case Key::Left:        return "Left";
        case Key::Right:       return "Right";
        case Key::One:         return "1";
        case Key::Two:         return "2";
        case Key::Three:       return "3";
        case Key::Four:        return "4";
        case Key::Five:        return "5";
        case Key::Six:         return "6";
        case Key::Seven:       return "7";
        case Key::Eight:       return "8";
        case Key::Nine:        return "9";
        case Key::MouseLeft:   return "MouseLeft";
        case Key::MouseRight:  return "MouseRight";
        case Key::MouseMiddle: return "MouseMiddle";
    }
    return "None";
}

Key keyFromString(const std::string& str) {
    // Reverse lookup — O(n) but called only during load, not per-frame
    static const std::unordered_map<std::string, Key> lookup = {
        {"None",        Key::None},
        {"Space",       Key::Space},
        {"LeftShift",   Key::LeftShift},
        {"LeftControl", Key::LeftControl},
        {"W",           Key::W},
        {"A",           Key::A},
        {"S",           Key::S},
        {"D",           Key::D},
        {"Q",           Key::Q},
        {"E",           Key::E},
        {"F",           Key::F},
        {"R",           Key::R},
        {"F1",          Key::F1},
        {"F2",          Key::F2},
        {"F3",          Key::F3},
        {"F4",          Key::F4},
        {"F5",          Key::F5},
        {"F6",          Key::F6},
        {"F7",          Key::F7},
        {"F8",          Key::F8},
        {"F9",          Key::F9},
        {"F10",         Key::F10},
        {"Escape",      Key::Escape},
        {"Tab",         Key::Tab},
        {"Up",          Key::Up},
        {"Down",        Key::Down},
        {"Left",        Key::Left},
        {"Right",       Key::Right},
        {"1",           Key::One},
        {"2",           Key::Two},
        {"3",           Key::Three},
        {"4",           Key::Four},
        {"5",           Key::Five},
        {"6",           Key::Six},
        {"7",           Key::Seven},
        {"8",           Key::Eight},
        {"9",           Key::Nine},
        {"MouseLeft",   Key::MouseLeft},
        {"MouseRight",  Key::MouseRight},
        {"MouseMiddle", Key::MouseMiddle},
    };

    auto it = lookup.find(str);
    if (it != lookup.end()) {
        return it->second;
    }
    return Key::None;
}

// ---------------------------------------------------------------------------
// InputState
// ---------------------------------------------------------------------------
bool InputState::isDown(Key key) const {
    auto it = keysDown.find(key);
    return it != keysDown.end() && it->second;
}

bool InputState::isJustPressed(Key key) const {
    auto it = keysJustPressed.find(key);
    return it != keysJustPressed.end() && it->second;
}

bool InputState::isJustReleased(Key key) const {
    auto it = keysJustReleased.find(key);
    return it != keysJustReleased.end() && it->second;
}

void InputState::update() {
    // Clear one-frame events (keysPressedForTick survives until a tick runs)
    keysJustPressed.clear();
    keysJustReleased.clear();
    mouseDelta = Vec2{0, 0};
    scrollDelta = 0.f;
}

bool InputState::isPressedForTick(Key key) const {
    auto it = keysPressedForTick.find(key);
    return it != keysPressedForTick.end() && it->second;
}

void InputState::clearTickPresses() {
    keysPressedForTick.clear();
}

void InputState::clearMouseDelta() {
    mouseDelta = Vec2{0.f, 0.f};
}

#ifdef __OBJC__

#import <Cocoa/Cocoa.h>

// ---------------------------------------------------------------------------
// Cocoa virtual key codes (from NSEvent documentation)
// ---------------------------------------------------------------------------
static constexpr NSInteger KEYCODE_SPACE       = 49;
static constexpr NSInteger KEYCODE_LEFTSHIFT   = 56;
static constexpr NSInteger KEYCODE_LEFTCONTROL = 59;
static constexpr NSInteger KEYCODE_W           = 13;
static constexpr NSInteger KEYCODE_A           = 0;
static constexpr NSInteger KEYCODE_S           = 1;
static constexpr NSInteger KEYCODE_D           = 2;
static constexpr NSInteger KEYCODE_Q           = 12;
static constexpr NSInteger KEYCODE_E           = 14;
static constexpr NSInteger KEYCODE_F           = 3;
static constexpr NSInteger KEYCODE_R           = 15;
static constexpr NSInteger KEYCODE_F1          = 122;
static constexpr NSInteger KEYCODE_F2          = 120;
static constexpr NSInteger KEYCODE_F3          = 99;
static constexpr NSInteger KEYCODE_F4          = 118;
static constexpr NSInteger KEYCODE_F5          = 96;
static constexpr NSInteger KEYCODE_F6          = 97;
static constexpr NSInteger KEYCODE_F7          = 98;
static constexpr NSInteger KEYCODE_F8          = 100;
static constexpr NSInteger KEYCODE_F9          = 101;
static constexpr NSInteger KEYCODE_F10         = 109;
static constexpr NSInteger KEYCODE_ESCAPE      = 53;
static constexpr NSInteger KEYCODE_TAB         = 48;
static constexpr NSInteger KEYCODE_UP          = 126;
static constexpr NSInteger KEYCODE_DOWN        = 125;
static constexpr NSInteger KEYCODE_LEFT        = 123;
static constexpr NSInteger KEYCODE_RIGHT       = 124;
static constexpr NSInteger KEYCODE_ONE         = 18;
static constexpr NSInteger KEYCODE_TWO         = 19;
static constexpr NSInteger KEYCODE_THREE       = 20;
static constexpr NSInteger KEYCODE_FOUR        = 21;
static constexpr NSInteger KEYCODE_FIVE        = 23;
static constexpr NSInteger KEYCODE_SIX         = 22;
static constexpr NSInteger KEYCODE_SEVEN       = 26;
static constexpr NSInteger KEYCODE_EIGHT       = 28;
static constexpr NSInteger KEYCODE_NINE        = 25;

// ---------------------------------------------------------------------------
// InputManager
// ---------------------------------------------------------------------------
InputManager::InputManager(NSWindow* window)
    : state_{}
    , window_(window)
    , lastMousePosition_{0, 0}
    , captured_(false)
    , cursorHidden_(false)
    , keyDownMonitor_(nil)
    , keyUpMonitor_(nil)
    , mouseMovedMonitor_(nil)
    , mouseDraggedMonitor_(nil)
    , mouseDownMonitor_(nil)
    , mouseUpMonitor_(nil)
    , scrollWheelMonitor_(nil) {
    assert(window_ != nil && "InputManager requires a non-nil window");

    // Cursor warps (capture/release) briefly suppress local mouse events by
    // default; zero the interval so menu hover works immediately.
    CGEventSourceRef eventSource = CGEventSourceCreate(kCGEventSourceStateCombinedSessionState);
    if (eventSource) {
        CGEventSourceSetLocalEventsSuppressionInterval(eventSource, 0.0);
        CFRelease(eventSource);
    }

    // Register local event monitors
    auto self = this;

    keyDownMonitor_ = [NSEvent addLocalMonitorForEventsMatchingMask:
        NSEventMaskKeyDown
        handler: ^NSEvent*(NSEvent* event) {
            self->handleKeyDown(event);
            return nil;  // consume event
        }];

    keyUpMonitor_ = [NSEvent addLocalMonitorForEventsMatchingMask:
        NSEventMaskKeyUp
        handler: ^NSEvent*(NSEvent* event) {
            self->handleKeyUp(event);
            return nil;
        }];

    mouseMovedMonitor_ = [NSEvent addLocalMonitorForEventsMatchingMask:
        NSEventMaskMouseMoved
        handler: ^NSEvent*(NSEvent* event) {
            self->handleMouseMoved(event);
            return event;  // pass through
        }];

    mouseDraggedMonitor_ = [NSEvent addLocalMonitorForEventsMatchingMask:
        NSEventMaskLeftMouseDragged | NSEventMaskRightMouseDragged | NSEventMaskOtherMouseDragged
        handler: ^NSEvent*(NSEvent* event) {
            self->handleMouseDragged(event);
            return event;
        }];

    mouseDownMonitor_ = [NSEvent addLocalMonitorForEventsMatchingMask:
        NSEventMaskLeftMouseDown | NSEventMaskRightMouseDown | NSEventMaskOtherMouseDown
        handler: ^NSEvent*(NSEvent* event) {
            self->handleMouseDown(event);
            return event;
        }];

    mouseUpMonitor_ = [NSEvent addLocalMonitorForEventsMatchingMask:
        NSEventMaskLeftMouseUp | NSEventMaskRightMouseUp | NSEventMaskOtherMouseUp
        handler: ^NSEvent*(NSEvent* event) {
            self->handleMouseUp(event);
            return event;
        }];

    scrollWheelMonitor_ = [NSEvent addLocalMonitorForEventsMatchingMask:
        NSEventMaskScrollWheel
        handler: ^NSEvent*(NSEvent* event) {
            self->handleScrollWheel(event);
            return event;
        }];

    // Enable tracking for mouse events
    [window_ setAcceptsMouseMovedEvents: true];
}

InputManager::~InputManager() {
    // Remove event monitors to prevent dangling references
    if (keyDownMonitor_)   [NSEvent removeMonitor: keyDownMonitor_];
    if (keyUpMonitor_)     [NSEvent removeMonitor: keyUpMonitor_];
    if (mouseMovedMonitor_)   [NSEvent removeMonitor: mouseMovedMonitor_];
    if (mouseDraggedMonitor_) [NSEvent removeMonitor: mouseDraggedMonitor_];
    if (mouseDownMonitor_)  [NSEvent removeMonitor: mouseDownMonitor_];
    if (mouseUpMonitor_)    [NSEvent removeMonitor: mouseUpMonitor_];
    if (scrollWheelMonitor_) [NSEvent removeMonitor: scrollWheelMonitor_];

    // Never leave the (global) mouse association broken behind us
    if (captured_) {
        CGAssociateMouseAndMouseCursorPosition(true);
    }
    if (cursorHidden_) {
        [NSCursor unhide];
    }
}

InputState& InputManager::state() {
    return state_;
}

void InputManager::handleKeyDown(NSEvent* event) {
    Key key = keyCodeToKey([event keyCode]);

    state_.keysJustPressed[key] = true;
    state_.keysPressedForTick[key] = true;
    state_.keysDown[key] = true;
}

void InputManager::handleKeyUp(NSEvent* event) {
    Key key = keyCodeToKey([event keyCode]);

    state_.keysJustReleased[key] = true;
    state_.keysDown[key] = false;
}

void InputManager::handleMouseMoved(NSEvent* event) {
    if (captured_) {
        // Raw hardware deltas, ACCUMULATED — multiple events can arrive per
        // frame and overwriting dropped motion. deltaY is positive downward;
        // the negation preserves the bottom-left-origin convention the
        // camera consumes (mouse up = +y = pitch up).
        state_.mouseDelta.x += static_cast<float>([event deltaX]);
        state_.mouseDelta.y -= static_cast<float>([event deltaY]);
        return;
    }
    NSPoint loc = [event locationInWindow];
    Vec2 current{static_cast<float>(loc.x), static_cast<float>(loc.y)};
    state_.mousePosition = current;
    lastMousePosition_ = current;
}

void InputManager::handleMouseDragged(NSEvent* event) {
    handleMouseMoved(event);
}

void InputManager::handleMouseDown(NSEvent* event) {
    NSInteger button = [event buttonNumber];
    if (button == 0) {
        state_.mouseLeftDown = true;
        state_.keysJustPressed[Key::MouseLeft] = true;
        state_.keysPressedForTick[Key::MouseLeft] = true;
        state_.keysDown[Key::MouseLeft] = true;
    } else if (button == 1) {
        state_.mouseRightDown = true;
        state_.keysJustPressed[Key::MouseRight] = true;
        state_.keysPressedForTick[Key::MouseRight] = true;
        state_.keysDown[Key::MouseRight] = true;
    } else if (button == 2) {
        state_.keysJustPressed[Key::MouseMiddle] = true;
        state_.keysPressedForTick[Key::MouseMiddle] = true;
        state_.keysDown[Key::MouseMiddle] = true;
    }
}

void InputManager::handleMouseUp(NSEvent* event) {
    NSInteger button = [event buttonNumber];
    if (button == 0) {
        state_.mouseLeftDown = false;
        state_.keysJustReleased[Key::MouseLeft] = true;
        state_.keysDown[Key::MouseLeft] = false;
    } else if (button == 1) {
        state_.mouseRightDown = false;
        state_.keysJustReleased[Key::MouseRight] = true;
        state_.keysDown[Key::MouseRight] = false;
    } else if (button == 2) {
        state_.keysJustReleased[Key::MouseMiddle] = true;
        state_.keysDown[Key::MouseMiddle] = false;
    }
}

void InputManager::handleScrollWheel(NSEvent* event) {
    state_.scrollDelta += static_cast<float>([event scrollingDeltaY]);
}

void InputManager::warpCursorToWindowCenter() {
    if (!window_) return;
    NSRect contentRect = [window_ contentRectForFrameRect:[window_ frame]];
    NSPoint centerCocoa{NSMidX(contentRect), NSMidY(contentRect)};
    // Cocoa screen coords are bottom-left origin; CG display coords are
    // top-left of the main display.
    CGFloat screenHeight = [[[NSScreen screens] firstObject] frame].size.height;
    CGWarpMouseCursorPosition(
        CGPointMake(centerCocoa.x, screenHeight - centerCocoa.y));

    // Keep the software position in sync with the warp
    NSRect windowContent = [window_ contentRectForFrameRect:[window_ frame]];
    lastMousePosition_ = Vec2{static_cast<float>(windowContent.size.width * 0.5),
                              static_cast<float>(windowContent.size.height * 0.5)};
    state_.mousePosition = lastMousePosition_;
}

void InputManager::captureMouse() {
    if (captured_) return;
    captured_ = true;

    warpCursorToWindowCenter();
    CGAssociateMouseAndMouseCursorPosition(false);
    if (!cursorHidden_) {
        [NSCursor hide];
        cursorHidden_ = true;
    }
    state_.clearMouseDelta();
}

void InputManager::releaseMouse() {
    if (!captured_) return;
    captured_ = false;

    // Warp BEFORE re-associating so the first hover events aren't suppressed
    warpCursorToWindowCenter();
    CGAssociateMouseAndMouseCursorPosition(true);
    if (cursorHidden_) {
        [NSCursor unhide];
        cursorHidden_ = false;
    }
    state_.clearMouseDelta();
}

Key InputManager::keyCodeToKey(NSInteger keyCode) {
    switch (keyCode) {
        case KEYCODE_SPACE:       return Key::Space;
        case KEYCODE_LEFTSHIFT:   return Key::LeftShift;
        case KEYCODE_LEFTCONTROL: return Key::LeftControl;
        case KEYCODE_W:           return Key::W;
        case KEYCODE_A:           return Key::A;
        case KEYCODE_S:           return Key::S;
        case KEYCODE_D:           return Key::D;
        case KEYCODE_Q:           return Key::Q;
        case KEYCODE_E:           return Key::E;
        case KEYCODE_F:           return Key::F;
        case KEYCODE_R:           return Key::R;
        case KEYCODE_F1:          return Key::F1;
        case KEYCODE_F2:          return Key::F2;
        case KEYCODE_F3:          return Key::F3;
        case KEYCODE_F4:          return Key::F4;
        case KEYCODE_F5:          return Key::F5;
        case KEYCODE_F6:          return Key::F6;
        case KEYCODE_F7:          return Key::F7;
        case KEYCODE_F8:          return Key::F8;
        case KEYCODE_F9:          return Key::F9;
        case KEYCODE_F10:         return Key::F10;
        case KEYCODE_ESCAPE:      return Key::Escape;
        case KEYCODE_TAB:         return Key::Tab;
        case KEYCODE_UP:          return Key::Up;
        case KEYCODE_DOWN:        return Key::Down;
        case KEYCODE_LEFT:        return Key::Left;
        case KEYCODE_RIGHT:       return Key::Right;
        case KEYCODE_ONE:         return Key::One;
        case KEYCODE_TWO:         return Key::Two;
        case KEYCODE_THREE:       return Key::Three;
        case KEYCODE_FOUR:        return Key::Four;
        case KEYCODE_FIVE:        return Key::Five;
        case KEYCODE_SIX:         return Key::Six;
        case KEYCODE_SEVEN:       return Key::Seven;
        case KEYCODE_EIGHT:       return Key::Eight;
        case KEYCODE_NINE:        return Key::Nine;
        default:                  return Key::None;
    }
}

#endif  // __OBJC__

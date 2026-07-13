#pragma once

#include <engine/input.hpp>

#include <array>
#include <optional>
#include <string>

// ---------------------------------------------------------------------------
// ActionBinding — Maps a single Key to a named game action
// ---------------------------------------------------------------------------
struct ActionBinding {
    Key key = Key::None;
    std::string name;
};

// ---------------------------------------------------------------------------
// InputBindings — Player-configurable key bindings
//
// Serialized as minimal JSON: {"forward": "W", "backward": "S", ...}
// Default path: ~/Library/Preferences/rycraft/bindings.json
// ---------------------------------------------------------------------------
struct InputBindings {
    ActionBinding forward = {Key::W, "Forward"};
    ActionBinding backward = {Key::S, "Backward"};
    ActionBinding left = {Key::A, "Left"};
    ActionBinding right = {Key::D, "Right"};
    ActionBinding jump = {Key::Space, "Jump"};
    ActionBinding sprint = {Key::LeftShift, "Sprint"};
    ActionBinding sneak = {Key::LeftControl, "Sneak"};
    ActionBinding inventory = {Key::E, "Inventory"};
    ActionBinding drop = {Key::Q, "Drop"};
    std::array<ActionBinding, 9> hotbar = {{
        {Key::One, "Slot 1"}, {Key::Two, "Slot 2"}, {Key::Three, "Slot 3"},
        {Key::Four, "Slot 4"}, {Key::Five, "Slot 5"},
        {Key::Six, "Slot 6"}, {Key::Seven, "Slot 7"},
        {Key::Eight, "Slot 8"}, {Key::Nine, "Slot 9"},
    }};

    // Returns false (with a logged error) when the file cannot be written.
    bool save(const std::string& path) const;

    // Returns nullopt only when the file exists but cannot be parsed; a
    // missing file yields the defaults (first launch is not an error).
    static std::optional<InputBindings> load(const std::string& path);

    static std::string defaultPath();
};

#include <engine/input_bindings.hpp>

#include <common/error.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers: minimal JSON serialization / deserialization
//
// Format: {"forward": "W", "backward": "S", ...}
// No external JSON library — simple string parsing.
// ---------------------------------------------------------------------------

// Create directories recursively (like mkdir -p)
static bool ensureDirectory(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        RY_LOG_ERROR(std::string("Failed to create directory: ") + path);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// InputBindings — Serialization
// ---------------------------------------------------------------------------

bool InputBindings::save(const std::string& path) const {
    // Ensure parent directory exists
    fs::path p(path);
    fs::path parent = p.parent_path();
    if (!parent.empty()) {
        if (!ensureDirectory(parent.string())) {
            RY_LOG_ERROR("Failed to create directory: " + parent.string());
            return false;
        }
    }

    std::ostringstream json;
    json << "{\n";

    auto writeBinding = [&](const std::string& name, Key key, bool last) {
        json << "  \"" << name << "\": \"" << keyToString(key) << "\"";
        if (!last) json << ",";
        json << "\n";
    };

    writeBinding("forward", forward.key, false);
    writeBinding("backward", backward.key, false);
    writeBinding("left", left.key, false);
    writeBinding("right", right.key, false);
    writeBinding("jump", jump.key, false);
    writeBinding("sprint", sprint.key, false);
    writeBinding("sneak", sneak.key, false);
    writeBinding("inventory", inventory.key, false);
    writeBinding("drop", drop.key, false);

    for (int i = 0; i < 9; ++i) {
        std::string slotName = "hotbar" + std::to_string(i);
        writeBinding(slotName, hotbar[i].key, (i == 8));
    }

    json << "}\n";

    std::ofstream file(path);
    if (!file.is_open()) {
        RY_LOG_ERROR("Failed to open file for writing: " + path);
        return false;
    }

    file << json.str();

    if (file.fail()) {
        file.close();
        RY_LOG_ERROR("Failed to write bindings file: " + path);
        return false;
    }

    file.close();
    return true;
}

std::optional<InputBindings> InputBindings::load(const std::string& path) {
    // Open file
    std::ifstream file(path);
    if (!file.is_open()) {
        // File not found — first launch, use defaults
        InputBindings defaults;
        return defaults;
    }

    // Read entire file into string
    std::ostringstream ss;
    ss << file.rdbuf();
    file.close();
    std::string content = ss.str();

    if (content.empty()) {
        RY_LOG_ERROR("Input bindings file is empty, using defaults");
        InputBindings defaults;
        return defaults;
    }

    // Minimal JSON parsing: find "key": "value" pairs
    InputBindings bindings;

    auto parseBinding = [&](ActionBinding& field, const std::string& jsonKey) -> bool {
        std::string searchKey = "\"" + jsonKey + "\"";
        size_t pos = content.find(searchKey);
        if (pos == std::string::npos) return false;

        // Find the colon after the key
        size_t colonPos = content.find(':', pos + searchKey.size());
        if (colonPos == std::string::npos) return false;

        // Find the value string (next quoted string after colon)
        size_t quoteStart = content.find('"', colonPos + 1);
        if (quoteStart == std::string::npos) return false;

        size_t quoteEnd = content.find('"', quoteStart + 1);
        if (quoteEnd == std::string::npos) return false;

        std::string value = content.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
        auto parsedKey = keyFromString(value);
        if (parsedKey == Key::None && !value.empty()) {
            RY_LOG_ERROR(std::string("Invalid binding value '") + value + "' for '" + jsonKey +
                         "', using default");
        }
        field.key = parsedKey;
        return true;
    };

    // If any binding fails to parse, falls back to defaults for that binding
    // (the defaults are already set in the struct constructor)
    parseBinding(bindings.forward, "forward");
    parseBinding(bindings.backward, "backward");
    parseBinding(bindings.left, "left");
    parseBinding(bindings.right, "right");
    parseBinding(bindings.jump, "jump");
    parseBinding(bindings.sprint, "sprint");
    parseBinding(bindings.sneak, "sneak");
    parseBinding(bindings.inventory, "inventory");
    parseBinding(bindings.drop, "drop");

    for (int i = 0; i < 9; ++i) {
        std::string slotName = "hotbar" + std::to_string(i);
        parseBinding(bindings.hotbar[i], slotName);
    }

    return bindings;
}

std::string InputBindings::defaultPath() {
    const char* home = getenv("HOME");
    if (!home) {
        // Fallback
        return "/tmp/rycraft_bindings.json";
    }
    return std::string(home) + "/Library/Preferences/rycraft/bindings.json";
}

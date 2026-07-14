#include "render/graphics_settings.hpp"

#include "common/error.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

// Minimal JSON of flat integer fields, matching the bindings-file idiom
// (no external JSON library). Booleans persist as 0/1.

namespace {

int clampInt(int v, int lo, int hi) {
    return std::clamp(v, lo, hi);
}

// "key": <int> — returns false when the key is absent or malformed.
bool parseIntField(const std::string& content, const char* key, int& out) {
    std::string searchKey = std::string("\"") + key + "\"";
    size_t pos = content.find(searchKey);
    if (pos == std::string::npos) {
        return false;
    }
    size_t colonPos = content.find(':', pos + searchKey.size());
    if (colonPos == std::string::npos) {
        return false;
    }
    const char* start = content.c_str() + colonPos + 1;
    char* end = nullptr;
    long value = std::strtol(start, &end, 10);
    if (end == start) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

// Environment integer override; leaves `out` untouched when unset/invalid.
// Returns true when the variable fired.
bool envInt(const char* name, int& out, int lo, int hi) {
    if (const char* value = std::getenv(name)) {
        if (*value) {
            out = clampInt(std::atoi(value), lo, hi);
            return true;
        }
    }
    return false;
}

bool envBool(const char* name, bool& out) {
    if (const char* value = std::getenv(name)) {
        if (*value) {
            out = std::atoi(value) != 0;
            return true;
        }
    }
    return false;
}

} // namespace

bool GraphicsSettings::applyEnvOverrides() {
    bool any = false;
    any |= envInt("RYCRAFT_SHADOWS", shadowQuality, 0, SHADOW_QUALITY_MAX);
    any |= envBool("RYCRAFT_VL", volumetricLight);
    any |= envInt("RYCRAFT_CLOUDS", cloudMode, 0, CLOUD_MODE_MAX);
    any |= envBool("RYCRAFT_SSAO", ssao);
    any |= envBool("RYCRAFT_SSR", waterReflections);
    any |= envBool("RYCRAFT_WAVING", wavingFoliage);
    any |= envBool("RYCRAFT_LENS_FLARE", lensFlare);
    any |= envInt("RYCRAFT_VIBRANCE", vibrance, 0, LEVEL_MAX);
    any |= envInt("RYCRAFT_SHARPEN", sharpening, 0, LEVEL_MAX);
    // RYCRAFT_BLOOM predates the settings system and speaks intensity
    // (0..1); invert the one shared factor so both controls stay one knob.
    if (const char* bloomEnv = std::getenv("RYCRAFT_BLOOM")) {
        if (*bloomEnv) {
            float intensity = std::clamp(static_cast<float>(std::atof(bloomEnv)), 0.f, 2.f);
            bloomLevel = clampInt(static_cast<int>(intensity / BLOOM_INTENSITY_PER_LEVEL + 0.5f), 0,
                                  LEVEL_MAX);
            any = true;
        }
    }
    return any;
}

bool saveSettings(const std::string& path, const SettingsValues& values,
                  const GraphicsSettings& gfx) {
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            RY_LOG_ERROR(("Failed to create settings directory: " + parent.string()).c_str());
            return false;
        }
    }

    std::ostringstream json;
    json << "{\n";
    auto writeField = [&](const char* key, int value, bool last = false) {
        json << "  \"" << key << "\": " << value << (last ? "\n" : ",\n");
    };
    writeField("viewDistance", values.viewDistance);
    writeField("fogLevel", values.fogLevel);
    writeField("sensitivityLevel", values.sensitivityLevel);
    writeField("volumeLevel", values.volumeLevel);
    writeField("shadowQuality", gfx.shadowQuality);
    writeField("volumetricLight", gfx.volumetricLight ? 1 : 0);
    writeField("cloudMode", gfx.cloudMode);
    writeField("ssao", gfx.ssao ? 1 : 0);
    writeField("waterReflections", gfx.waterReflections ? 1 : 0);
    writeField("wavingFoliage", gfx.wavingFoliage ? 1 : 0);
    writeField("lensFlare", gfx.lensFlare ? 1 : 0);
    writeField("bloomLevel", gfx.bloomLevel);
    writeField("vibrance", gfx.vibrance);
    writeField("sharpening", gfx.sharpening, true);
    json << "}\n";

    std::ofstream file(path);
    if (!file.is_open()) {
        RY_LOG_ERROR(("Failed to open settings file for writing: " + path).c_str());
        return false;
    }
    file << json.str();
    if (file.fail()) {
        RY_LOG_ERROR(("Failed to write settings file: " + path).c_str());
        return false;
    }
    return true;
}

LoadedSettings loadSettings(const std::string& path) {
    LoadedSettings loaded; // field defaults survive missing keys/file

    std::ifstream file(path);
    if (!file.is_open()) {
        return loaded; // first launch
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();

    int v = 0;
    if (parseIntField(content, "viewDistance", v)) {
        loaded.values.viewDistance = clampInt(v, 4, 32);
    }
    if (parseIntField(content, "fogLevel", v)) {
        loaded.values.fogLevel = clampInt(v, 0, 10);
    }
    if (parseIntField(content, "sensitivityLevel", v)) {
        loaded.values.sensitivityLevel = clampInt(v, 1, 10);
    }
    if (parseIntField(content, "volumeLevel", v)) {
        loaded.values.volumeLevel = clampInt(v, 0, 10);
    }
    if (parseIntField(content, "shadowQuality", v)) {
        loaded.gfx.shadowQuality = clampInt(v, 0, GraphicsSettings::SHADOW_QUALITY_MAX);
    }
    if (parseIntField(content, "volumetricLight", v)) {
        loaded.gfx.volumetricLight = v != 0;
    }
    if (parseIntField(content, "cloudMode", v)) {
        loaded.gfx.cloudMode = clampInt(v, 0, GraphicsSettings::CLOUD_MODE_MAX);
    }
    if (parseIntField(content, "ssao", v)) {
        loaded.gfx.ssao = v != 0;
    }
    if (parseIntField(content, "waterReflections", v)) {
        loaded.gfx.waterReflections = v != 0;
    }
    if (parseIntField(content, "wavingFoliage", v)) {
        loaded.gfx.wavingFoliage = v != 0;
    }
    if (parseIntField(content, "lensFlare", v)) {
        loaded.gfx.lensFlare = v != 0;
    }
    if (parseIntField(content, "bloomLevel", v)) {
        loaded.gfx.bloomLevel = clampInt(v, 0, GraphicsSettings::LEVEL_MAX);
    }
    if (parseIntField(content, "vibrance", v)) {
        loaded.gfx.vibrance = clampInt(v, 0, GraphicsSettings::LEVEL_MAX);
    }
    if (parseIntField(content, "sharpening", v)) {
        loaded.gfx.sharpening = clampInt(v, 0, GraphicsSettings::LEVEL_MAX);
    }
    return loaded;
}

std::string settingsPath() {
    const char* home = getenv("HOME");
    if (!home) {
        return "/tmp/rycraft_settings.json";
    }
    return std::string(home) + "/Library/Preferences/rycraft/settings.json";
}

#pragma once

#include "render/ui_menu.hpp"

#include <string>

// ---------------------------------------------------------------------------
// GraphicsSettings — the video screen's per-effect toggles and levels.
//
// Pure C++ (no Metal) so layouts and persistence are unit-testable. Defaults
// are the maximum-quality preset: every effect on. The engine owns the live
// instance and pushes copies to the render pipeline; each effect's pass
// reads its own field as it lands in the overhaul.
// ---------------------------------------------------------------------------
struct GraphicsSettings {
    // Field bounds, shared by the load clamp, the env clamp, and the menu
    // steppers so no path can produce a value another rejects (the
    // SHADOW/CLOUD name tables index by these).
    static constexpr int SHADOW_QUALITY_MAX = 2;
    static constexpr int CLOUD_MODE_MAX = 2;
    static constexpr int LEVEL_MAX = 10;

    // bloomLevel ↔ bloom intensity: level 5 = stock strength 1.0. The env
    // parser (legacy RYCRAFT_BLOOM speaks intensity) inverts this same
    // factor so the two mappings can never drift apart.
    static constexpr float BLOOM_INTENSITY_PER_LEVEL = 0.2f;

    int shadowQuality = 2;        // 0 off, 1 medium (2×1024²), 2 high (3×2048²)
    bool volumetricLight = true;  // ray-marched sun/moon light shafts
    int cloudMode = 2;            // 0 off, 1 flat plane layer, 2 volumetric
    bool ssao = true;             // half-res screen-space ambient occlusion
    bool waterReflections = true; // SSR on water; sky fresnel stays regardless
    bool wavingFoliage = true;    // wind sway on grass/flora/leaves
    bool lensFlare = true;        // occlusion-tested sun flare
    int bloomLevel = 5;           // 0-10; 0 skips the bloom passes entirely
    int vibrance = 5;             // 0-10 color-grade strength; 5 = stock look
    int sharpening = 0;           // 0-10 CAS strength; 0 skips (no TAA blur to undo)

    float bloomIntensity() const {
        return static_cast<float>(bloomLevel) * BLOOM_INTENSITY_PER_LEVEL;
    }

    // Headless-playtest overrides (RYCRAFT_SHADOWS, RYCRAFT_VL, RYCRAFT_CLOUDS,
    // RYCRAFT_SSAO, RYCRAFT_SSR, RYCRAFT_WAVING, RYCRAFT_LENS_FLARE,
    // RYCRAFT_BLOOM, RYCRAFT_VIBRANCE, RYCRAFT_SHARPEN). Applied after load().
    // Returns true when any override fired — the engine then skips every
    // settings save so a playtest env never rewrites the user's file.
    bool applyEnvOverrides();
};

// The general values and the video settings persist together in ONE
// settings.json — a second file would be a second source of truth for
// "the settings". Missing file or missing keys keep the field defaults.
struct LoadedSettings {
    SettingsValues values;
    GraphicsSettings gfx;
};

bool saveSettings(const std::string& path, const SettingsValues& values,
                  const GraphicsSettings& gfx);
LoadedSettings loadSettings(const std::string& path);
std::string settingsPath(); // ~/Library/Preferences/rycraft/settings.json

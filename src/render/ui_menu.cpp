#include "render/ui_menu.hpp"

#include <array>

// Font metrics mirrored from UIOverlay (8×8 glyphs, 1px advance gap)
static constexpr float FONT_CELL = 9.0f;
static constexpr float FONT_HEIGHT = 8.0f;

float menuTextWidth(const std::string& text, float scale, float pixelWidth) {
    return static_cast<float>(text.size()) * FONT_CELL * scale / pixelWidth;
}

namespace {

// Shared building blocks, all sized against a 768-pt-tall reference window.
struct LayoutContext {
    float w, h;      // window size in points
    float s;         // reference scale: 1.0 at 768-pt-tall

    float px(float v) const { return v * s / w; }  // ref-pixels → normalized X
    float py(float v) const { return v * s / h; }  // ref-pixels → normalized Y
};

void addCenteredText(MenuLayout& layout, const LayoutContext& ctx, const std::string& text,
                     float centerY, float scale, float r = 1.f, float g = 1.f, float b = 1.f) {
    float width = menuTextWidth(text, scale * ctx.s, ctx.w);
    layout.texts.push_back(MenuText{0.5f - width * 0.5f,
                                    centerY - ctx.py(FONT_HEIGHT * scale) * 0.5f,
                                    scale * ctx.s, text, r, g, b});
}

void addButton(MenuLayout& layout, const LayoutContext& ctx, const std::string& label,
               MenuAction action, float centerX, float centerY, float refW, float refH) {
    UIRect rect{centerX - ctx.px(refW) * 0.5f, centerY - ctx.py(refH) * 0.5f,
                ctx.px(refW), ctx.py(refH)};
    layout.buttons.push_back(MenuButton{rect, label, action});
}

// One settings row: label on the left, [-] value [+] on the right.
void addSettingsRow(MenuLayout& layout, const LayoutContext& ctx, const std::string& label,
                    const std::string& value, MenuAction down, MenuAction up, float centerY) {
    const float labelScale = 2.0f;
    layout.texts.push_back(MenuText{0.5f - ctx.px(250.f),
                                    centerY - ctx.py(FONT_HEIGHT * labelScale) * 0.5f,
                                    labelScale * ctx.s, label});

    const float stepperX = 0.5f + ctx.px(120.f);
    addButton(layout, ctx, "-", down, stepperX - ctx.px(70.f), centerY, 32.f, 32.f);
    float valueWidth = menuTextWidth(value, labelScale * ctx.s, ctx.w);
    layout.texts.push_back(MenuText{stepperX - valueWidth * 0.5f,
                                    centerY - ctx.py(FONT_HEIGHT * labelScale) * 0.5f,
                                    labelScale * ctx.s, value, 1.f, 0.9f, 0.4f});
    addButton(layout, ctx, "+", up, stepperX + ctx.px(70.f), centerY, 32.f, 32.f);
}

MenuLayout buildTitleLayout(const LayoutContext& ctx) {
    MenuLayout layout;
    layout.dimAlpha = 0.25f;

    addCenteredText(layout, ctx, "rycraft", 0.68f, 8.0f);
    addCenteredText(layout, ctx, "A voxel world built on Metal", 0.58f, 1.5f, 0.85f, 0.85f, 0.9f);

    addButton(layout, ctx, "PLAY", MenuAction::Play, 0.5f, 0.42f, 320.f, 48.f);
    addButton(layout, ctx, "QUIT", MenuAction::Quit, 0.5f, 0.42f - ctx.py(68.f), 320.f, 48.f);
    return layout;
}

MenuLayout buildPauseLayout(const LayoutContext& ctx) {
    MenuLayout layout;
    layout.dimAlpha = 0.45f;
    layout.panel = UIRect{0.5f - ctx.px(210.f), 0.5f - ctx.py(170.f), ctx.px(420.f), ctx.py(340.f)};

    addCenteredText(layout, ctx, "PAUSED", 0.5f + ctx.py(120.f), 3.0f);

    addButton(layout, ctx, "RESUME", MenuAction::Resume, 0.5f, 0.5f + ctx.py(40.f), 320.f, 44.f);
    addButton(layout, ctx, "SETTINGS", MenuAction::OpenSettings, 0.5f, 0.5f - ctx.py(24.f), 320.f, 44.f);
    addButton(layout, ctx, "QUIT", MenuAction::Quit, 0.5f, 0.5f - ctx.py(88.f), 320.f, 44.f);
    return layout;
}

MenuLayout buildSettingsLayout(const LayoutContext& ctx, const SettingsValues& values) {
    MenuLayout layout;
    layout.dimAlpha = 0.45f;
    layout.panel = UIRect{0.5f - ctx.px(300.f), 0.5f - ctx.py(210.f), ctx.px(600.f), ctx.py(420.f)};

    addCenteredText(layout, ctx, "SETTINGS", 0.5f + ctx.py(160.f), 3.0f);

    addSettingsRow(layout, ctx, "RENDER DIST", std::to_string(values.viewDistance),
                   MenuAction::ViewDistanceDown, MenuAction::ViewDistanceUp,
                   0.5f + ctx.py(84.f));
    addSettingsRow(layout, ctx, "FOG", std::to_string(values.fogLevel),
                   MenuAction::FogDown, MenuAction::FogUp, 0.5f + ctx.py(28.f));
    addSettingsRow(layout, ctx, "SENSITIVITY", std::to_string(values.sensitivityLevel),
                   MenuAction::SensitivityDown, MenuAction::SensitivityUp,
                   0.5f - ctx.py(28.f));
    addSettingsRow(layout, ctx, "VOLUME", std::to_string(values.volumeLevel),
                   MenuAction::VolumeDown, MenuAction::VolumeUp, 0.5f - ctx.py(84.f));

    addButton(layout, ctx, "BACK", MenuAction::CloseSettings, 0.5f, 0.5f - ctx.py(160.f), 320.f, 44.f);
    return layout;
}

} // namespace

MenuLayout buildMenuLayout(GameScreen screen, float pixelWidth, float pixelHeight,
                           const SettingsValues& values) {
    if (pixelWidth <= 0.f || pixelHeight <= 0.f) return {};
    LayoutContext ctx{pixelWidth, pixelHeight, pixelHeight / 768.0f};

    switch (screen) {
        case GameScreen::Title:
            return buildTitleLayout(ctx);
        case GameScreen::Paused:
            return buildPauseLayout(ctx);
        case GameScreen::Settings:
            return buildSettingsLayout(ctx, values);
        case GameScreen::Playing:
            return {};
    }
    return {};
}

int menuHitTest(const MenuLayout& layout, float mouseX, float mouseY) {
    for (size_t i = 0; i < layout.buttons.size(); ++i) {
        if (layout.buttons[i].rect.contains(mouseX, mouseY)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

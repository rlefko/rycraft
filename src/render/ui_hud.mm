#include "render/ui_hud.hpp"

#include "render/ui_overlay.hpp"
#include "world/item.hpp"

void drawGameHud(UIOverlay& ui, const UIFrameState& frame, uint32_t displayWidth,
                 uint32_t displayHeight) {
    if (frame.screen == GameScreen::TITLE)
        return;

    const float w = static_cast<float>(displayWidth);
    const float h = static_cast<float>(displayHeight);

    // ---- Debug HUD (F3) ----
    if (frame.showDebugHud) {
        ui.drawPerformanceHUD(frame.stats);
    }

    // ---- Crosshair (gameplay only — menus have a real cursor) ----
    if (frame.screen == GameScreen::PLAYING) {
        float crossH = 2.0f / h;
        float crossW = 24.0f / w;
        float crossV = 24.0f / h;
        float crossLineW = 2.0f / w;
        ui.drawQuad(0.5f - crossW * 0.5f, 0.5f - crossH * 0.5f, crossW, crossH, 1.0f, 1.0f, 1.0f,
                    0.9f);
        ui.drawQuad(0.5f - crossLineW * 0.5f, 0.5f - crossV * 0.5f, crossLineW, crossV, 1.0f, 1.0f,
                    1.0f, 0.9f);
    }

    // ---- Hotbar (9 slots at bottom of screen) ----
    const int slotCount = static_cast<int>(frame.hotbar.slots.size());
    float slotSize = 48.0f / h;
    float slotGap = 2.0f / h;
    float hotbarY = 6.0f / h;
    float totalWidth = slotCount * slotSize + (slotCount - 1) * slotGap;
    float hotbarX = (1.0f - totalWidth) * 0.5f;

    int selectedIndex = frame.hotbar.selected;

    for (int i = 0; i < slotCount; ++i) {
        float slotX = hotbarX + i * (slotSize + slotGap);

        if (i == selectedIndex) {
            ui.drawQuad(slotX - 2.0f / w, hotbarY - 2.0f / h, slotSize + 4.0f / w,
                        slotSize + 4.0f / h, 1.0f, 1.0f, 1.0f, 0.8f);
        }

        ui.drawQuad(slotX, hotbarY, slotSize, slotSize, 0.3f, 0.3f, 0.3f, 0.6f);

        // Item indicator: the registry swatch color, empty slots stay bare
        const ItemStack& stack = frame.hotbar.slots[static_cast<size_t>(i)];
        if (stack.empty())
            continue;
        const uint32_t swatch = itemSwatchColor(stack.type);
        const float r = static_cast<float>((swatch >> 16) & 0xFF) / 255.0f;
        const float g = static_cast<float>((swatch >> 8) & 0xFF) / 255.0f;
        const float b = static_cast<float>(swatch & 0xFF) / 255.0f;

        float innerSize = slotSize * 0.7f;
        float innerOffset = (slotSize - innerSize) * 0.5f;
        ui.drawQuad(slotX + innerOffset, hotbarY + innerOffset, innerSize, innerSize, r, g, b,
                    0.9f);
    }
}

void drawMenu(UIOverlay& ui, const MenuLayout& layout, int hoveredButton, uint32_t displayWidth,
              uint32_t displayHeight) {
    const float w = static_cast<float>(displayWidth);
    const float h = static_cast<float>(displayHeight);

    if (layout.dimAlpha > 0.f) {
        ui.drawQuad(0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, layout.dimAlpha);
    }

    if (layout.panel.w > 0.f) {
        // Border quad behind the panel gives a cheap Minecraft-style bevel
        float bx = 3.0f / w;
        float by = 3.0f / h;
        ui.drawQuad(layout.panel.x - bx, layout.panel.y - by, layout.panel.w + 2 * bx,
                    layout.panel.h + 2 * by, 0.02f, 0.02f, 0.03f, 0.95f);
        ui.drawQuad(layout.panel.x, layout.panel.y, layout.panel.w, layout.panel.h, 0.11f, 0.11f,
                    0.14f, 0.92f);
    }

    for (size_t i = 0; i < layout.buttons.size(); ++i) {
        const MenuButton& button = layout.buttons[i];
        const bool hovered = static_cast<int>(i) == hoveredButton;

        float bx = 2.0f / w;
        float by = 2.0f / h;
        ui.drawQuad(button.rect.x - bx, button.rect.y - by, button.rect.w + 2 * bx,
                    button.rect.h + 2 * by, 0.02f, 0.02f, 0.03f, 0.95f);
        if (hovered) {
            ui.drawQuad(button.rect.x, button.rect.y, button.rect.w, button.rect.h, 0.45f, 0.52f,
                        0.82f, 0.95f);
        } else {
            ui.drawQuad(button.rect.x, button.rect.y, button.rect.w, button.rect.h, 0.33f, 0.33f,
                        0.38f, 0.92f);
        }

        const float labelScale = 2.0f * (h / 768.0f);
        float labelWidth = ui.measureString(button.label.c_str(), labelScale);
        float labelHeight = 8.0f * labelScale / h;
        float labelX = button.rect.x + (button.rect.w - labelWidth) * 0.5f;
        float labelY = button.rect.y + (button.rect.h - labelHeight) * 0.5f;
        float brightness = hovered ? 1.0f : 0.92f;
        ui.drawString(button.label.c_str(), labelX, labelY, labelScale, brightness, brightness,
                      brightness);
    }

    for (const MenuText& text : layout.texts) {
        ui.drawString(text.text.c_str(), text.x, text.y, text.scale, text.r, text.g, text.b);
    }
}

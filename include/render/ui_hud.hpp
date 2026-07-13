#pragma once

#import <Metal/Metal.h>

#include "engine/hotbar.hpp"
#include "render/ui_menu.hpp"

class UIOverlay;

// ---------------------------------------------------------------------------
// HUD + menu composition — everything the UI pass draws each frame.
//
// Lives outside RenderPipeline so screen composition (what the HUD shows,
// how menus look) evolves without touching the Metal pass plumbing. All
// drawing goes through UIOverlay's batched quad/text API between the
// caller's beginFrame()/flush().
// ---------------------------------------------------------------------------

// Gameplay chrome: crosshair (only while playing), hotbar, and the F3 debug
// HUD. Skipped entirely on the title screen.
void drawGameHud(UIOverlay& ui, const Hotbar& hotbar, const UIFrameState& frame,
                 uint32_t displayWidth, uint32_t displayHeight);

// The current menu (title/pause/settings), with hover highlighting.
void drawMenu(UIOverlay& ui, const MenuLayout& layout, int hoveredButton, uint32_t displayWidth,
              uint32_t displayHeight);

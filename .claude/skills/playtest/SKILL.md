---
name: playtest
description: Build and run rycraft with Metal validation, capture real frames as PNGs, and report what the game actually shows. Use to verify any player-visible change (rendering, UI, menus, world content), when the user asks to "run the game", "playtest", or "screenshot it", or as the verification step the render-review skill requires. Works headlessly — the engine writes its own frame captures, so no screen-recording permission is needed.
---

# Playtest

Drive the real game and report what it does, with frames to prove it.

## Step 1: Build

```bash
ninja -C build
```

Fail fast on compile errors — report them and stop. If `build/` doesn't exist: `meson setup build` first.

## Step 2: Run with validation and capture

The engine has first-class playtest hooks (no macOS screen-recording permission involved — it writes the PNG itself):

| Variable | Effect |
|----------|--------|
| `RYCRAFT_CAPTURE=/tmp/frame.png` | write one frame to that path as PNG |
| `RYCRAFT_CAPTURE_FRAME=N` | which frame to capture (default 240; world gen takes ~5–8 s, so 400–600 shows settled terrain) |
| `RYCRAFT_START_SCREEN=title\|playing\|paused\|settings` | start on a specific screen (menus render over the live world) |
| `RYCRAFT_BLOOM=0..1` | scale or disable bloom to isolate post-processing |

Standard invocation (run from a scratch directory — saves land in `./rycraft_world/`):

```bash
cd /tmp/rycraft-playtest && mkdir -p . && \
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog MTL_SHADER_VALIDATION=1 \
RYCRAFT_CAPTURE=/tmp/playtest.png RYCRAFT_CAPTURE_FRAME=500 \
<repo>/build/src/rycraft > /tmp/playtest.log 2>&1 &
sleep 20 && kill %1
```

Capture the screens the change touches (e.g. `paused` and `settings` for a menu change; `playing` at two different capture frames for world/lighting changes).

## Step 3: Inspect

1. **The log:** any `[MTLDebug]`/validation line is a failure to report verbatim. Also check for `[ERROR]`/`[FATAL]` engine lines and that the "Render: N loaded chunks" heartbeat advances (frozen frame counts mean a stalled loop).
2. **The frames:** Read the PNGs and actually look. Expected baseline while playing: sky gradient with clouds, textured terrain filling the lower frame with skylight shading, crosshair + hotbar; menus render sharp bitmap text over a dimmed world.
3. Kill any leftover `rycraft` process before finishing.

## Step 4: Report

1. **Verdict**: works as intended / broken / works with concerns.
2. **Evidence**: validation-message count, notable log lines, and what each captured frame shows relative to the change under test.
3. **Anything unverifiable headlessly** (e.g. sound output, feel of mouse input) — name it explicitly so a human tries it, rather than implying it was checked.

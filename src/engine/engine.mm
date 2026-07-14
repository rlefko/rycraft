#import <engine/engine.hpp>

#import <QuartzCore/QuartzCore.h>

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <common/error.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <engine/camera.hpp>
#include <engine/game_state.hpp>
#include <engine/hotbar.hpp>
#include <engine/input_bindings.hpp>
#include <entity/ai.hpp>
#include <entity/player.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/graphics_settings.hpp>
#include <render/render_pipeline.hpp>
#include <render/ui_menu.hpp>
#include <world/save_manager.hpp>
#include <world/world.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

// ---------------------------------------------------------------------------
// Engine — Singleton (Objective-C class with C++ internals)
// ---------------------------------------------------------------------------

// Raycast hit result type for block interaction
using BlockRayHit = std::optional<std::pair<Vec3, Vec3>>;

// Internal C++ state
struct EngineState {
    // ---- Game loop state ----
    double lastTime = 0;
    double deltaTime = 0;
    uint64_t frameCount = 0;

    // ---- Fixed timestep accumulator ----
    static constexpr double TICK_RATE = 20.0;
    static constexpr double TICK_DT = 1.0 / TICK_RATE;
    double accumulator = 0;

    // ---- Projection ----
    Mat4 projectionMatrix = Mat4::identity();
    CGSize drawableSize = {0, 0};

    // ---- Field of view ----
    // Sprinting widens the FOV for a feeling of speed; the value eases
    // per frame (render parameter, not sim state) toward the mode's target.
    static constexpr float BASE_FOV = 70.0f;
    static constexpr float SPRINT_FOV = 77.0f; // +10%, the classic sprint cue
    static constexpr float FOV_EASE_SECONDS = 0.1f;
    float fovCurrent = BASE_FOV;

    // ---- Day/Night Cycle & Weather ----
    static constexpr uint64_t TICKS_PER_DAY = 24000; // 20 min at 20Hz
    uint64_t worldTime = 0;
    bool raining = false; // deterministic per-day schedule (seeded)
    float wetness = 0.0f; // 0 dry .. 1 soaked; ramps with the rain
    // Weather tuning: how many days rain (percent), and how fast surfaces
    // soak under rain / dry out after it stops.
    static constexpr uint32_t RAIN_DAYS_PERCENT = 40;
    static constexpr float SOAK_SECONDS = 15.0f;
    static constexpr float DRY_SECONDS = 45.0f;

    // ---- Block Interaction ----
    Hotbar hotbar;
    Vec3 highlightedBlock; // Block currently targeted by crosshair
    bool hasHighlightedBlock = false;

    // ---- Game flow & UI ----
    GameFlow flow;                   // Title → Playing ⇄ Paused ⇄ Settings
    bool spawnValidated = false;     // player unstuck from stale-save terrain
    SettingsValues settings;         // live values shown in the settings menu
    GraphicsSettings gfx;            // video screen values (persisted with settings)
    bool envOverridesActive = false; // RYCRAFT_* session: never save settings
    MenuLayout menuLayout;           // rebuilt each frame while a menu is open
    int hoveredButton = -1;
    bool showDebugHud = false;

    // ---- Performance stats (exponential moving averages) ----
    float smoothedFrameMs = 16.7f;
    uint32_t cachedChunkCount = 0;
    uint32_t cachedPendingChunks = 0;

    // ---- Player & World ----
    Player player;
    std::shared_ptr<World> world;
    std::unique_ptr<SaveManager> saveManager;
    Camera camera;

    // ---- Animals ----
    std::unique_ptr<Spawner> spawner;
    std::unordered_map<uint64_t, StateMachine> entityBrains;
    bool populationSpawned = false;
    int animalCallCooldown = 0;

    // ---- Audio ----
    std::unique_ptr<AudioEngine> audio;
    std::vector<float> sfxBlockBreak;
    std::vector<float> sfxBlockPlace;
    std::vector<float> sfxFootstep;
    std::vector<float> sfxWind;
    std::vector<float> sfxAnimal[4]; // indexed by EntityType
    int32_t windVoice = -1;
    float footstepDistance = 0.f; // ground distance walked since last step
    Vec3 lastFootstepPos{0.f, 0.f, 0.f};

    // ---- Input manager (set after window creation) ----
    std::unique_ptr<InputManager> inputManager;
};

@implementation Engine {
    NSApplication* _app;
    NSWindow* _window;
    MTKView* _view;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;

    // Render pipeline (created after device/queue + shader library)
    std::unique_ptr<RenderPipeline> _renderPipeline;

    // C++ game state
    std::unique_ptr<EngineState> _state;

    // Scroll wheel tracking
    float _scrollAccumulator;

    // Save-on-quit guard (terminate can be reached from several paths)
    bool _savedWorld;
}

// ---- C++ bridge helper — must be inside @implementation to access _state ----
static EngineState* _engineGetState(Engine* engine) {
    return engine->_state.get();
}

+ (instancetype)sharedEngine {
    static Engine* shared = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        shared = [[Engine alloc] init];
    });
    return shared;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _app = nil;
        _window = nil;
        _view = nil;
        _device = nil;
        _queue = nil;
        _scrollAccumulator = 0;
        _state = std::make_unique<EngineState>();
        _state->saveManager = std::make_unique<SaveManager>([@"rycraft_world" UTF8String]);

        // Resume the saved world when one exists; otherwise start fresh
        uint32_t seed = 42;
        Vec3 spawnPos{0.f, 100.f, 0.f};
        if (auto meta = _state->saveManager->loadMetadata()) {
            seed = meta->seed;
            spawnPos = meta->spawnPos;
            _state->worldTime = meta->worldTime;
        }
        // Playtest hook: pin the time of day (0..23999; 6000 = noon).
        if (const char* timeEnv = std::getenv("RYCRAFT_TIME")) {
            _state->worldTime = static_cast<uint64_t>(std::clamp(std::atoi(timeEnv), 0, 23999));
        }

        // Persisted settings load before the World exists (view distance
        // feeds its constructor); env overrides win over the file for
        // headless playtests. Playtest hook: RYCRAFT_VIEW_DISTANCE=<4..32>.
        // An env-overridden session never saves settings — a playtest run
        // must not rewrite the user's file with its overrides.
        LoadedSettings loaded = loadSettings(settingsPath());
        _state->settings = loaded.values;
        _state->gfx = loaded.gfx;
        _state->envOverridesActive = _state->gfx.applyEnvOverrides();
        if (const char* vdEnv = std::getenv("RYCRAFT_VIEW_DISTANCE")) {
            _state->settings.viewDistance = std::clamp(std::atoi(vdEnv), 4, 32);
            _state->envOverridesActive = true;
        }
        _state->world = std::make_shared<World>(seed, _state->settings.viewDistance);
        // Chunks load from disk before regenerating, so block edits persist
        _state->world->setSaveManager(_state->saveManager.get());
        _state->spawner = std::make_unique<Spawner>(*_state->world);
        _state->player.position = spawnPos;
    }
    return self;
}

- (BOOL)initialize {
    // 1. Create NSApplication
    _app = [NSApplication sharedApplication];
    if (!_app) {
        RY_LOG_FATAL("Failed to create NSApplication");
        return NO;
    }
    [_app setActivationPolicy:NSApplicationActivationPolicyRegular];
    [_app activateIgnoringOtherApps:true];
    [_app setDelegate:self];

    // 2. Create Metal device
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        RY_LOG_FATAL("No Metal-capable device found — Metal is not supported on this hardware");
        return NO;
    }

    // 3. Create command queue
    _queue = [_device newCommandQueue];
    if (!_queue) {
        RY_LOG_FATAL("Failed to create Metal command queue");
        return NO;
    }

    // 4. Create NSWindow
    NSRect screenFrame = [[NSScreen mainScreen] visibleFrame];
    NSRect windowRect = NSMakeRect((screenFrame.size.width - 1024) * 0.5,
                                   (screenFrame.size.height - 768) * 0.5, 1024, 768);
    _window = [[NSWindow alloc]
        initWithContentRect:windowRect
                  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                            NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                    backing:NSBackingStoreBuffered
                      defer:false];
    if (!_window) {
        RY_LOG_FATAL("Failed to create NSWindow");
        return NO;
    }
    [_window setTitle:@"rycraft"];
    [_window setDelegate:self];
    [_window makeKeyAndOrderFront:nil];

    // 5. Create and configure MTKView
    _view = [[MTKView alloc] initWithFrame:windowRect device:_device];
    if (!_view) {
        RY_LOG_FATAL("Failed to create MTKView");
        return NO;
    }

    // Drawable pixel format. The render pipeline builds its own MSAA render
    // passes, so the view carries no sample count or depth buffer of its own.
    _view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;

    // Disable automatic setNeedsDisplay — we drive rendering from the game loop
    _view.enableSetNeedsDisplay = false;
    _view.framebufferOnly = false;

    // Delegate
    _view.delegate = self;

    // Set as window content
    [_window setContentView:_view];

    // 7. Create InputManager (the game opens on the title screen with a
    // free cursor; clicking PLAY captures the mouse)
    _state->inputManager = std::make_unique<InputManager>(_window);

    // 6. Load shader library and create render pipeline
    NSString* exePath = [[NSBundle mainBundle] executablePath];
    NSString* dirPath = [exePath stringByDeletingLastPathComponent];
    NSString* libPath = [dirPath stringByAppendingPathComponent:@"pipeline.metallib"];
    NSURL* libURL = [NSURL fileURLWithPath:libPath];
    NSError* libError = nil;
    id<MTLLibrary> library = [_device newLibraryWithURL:libURL error:&libError];
    if (libError) {
        NSString* msg = [NSString
            stringWithFormat:@"Failed to load shader library: %@", libError.localizedDescription];
        RY_LOG_FATAL([msg UTF8String]);
    }

    _renderPipeline = std::make_unique<RenderPipeline>(
        _device, library, static_cast<uint32_t>(_view.bounds.size.width),
        static_cast<uint32_t>(_view.bounds.size.height));

    // Apply the persisted settings to the live systems (world view distance
    // was already applied through the World constructor; RYCRAFT_BLOOM rides
    // GraphicsSettings::applyEnvOverrides now).
    _renderPipeline->setGraphicsSettings(_state->gfx);
    _renderPipeline->setFogDensity(fogDensityForLevel(_state->settings.fogLevel));
    _state->camera.setMouseSensitivity(mouseSensitivityForLevel(_state->settings.sensitivityLevel));

    // Playtest override: start on a specific screen
    // (title|playing|paused|settings|video)
    if (const char* screenEnv = std::getenv("RYCRAFT_START_SCREEN")) {
        std::string name = screenEnv;
        if (name == "playing") {
            _state->flow.screen = GameScreen::PLAYING;
            _state->inputManager->captureMouse();
        } else if (name == "paused") {
            _state->flow.screen = GameScreen::PAUSED;
        } else if (name == "settings") {
            _state->flow.screen = GameScreen::SETTINGS;
        } else if (name == "video") {
            _state->flow.screen = GameScreen::VIDEO_SETTINGS;
        }
    }

    // 8. Audio: non-fatal on failure — the game is fully playable silent
    _state->audio = std::make_unique<AudioEngine>();
    if (_state->audio->initialize()) {
        _state->sfxBlockBreak = SoundEffect::generateBlockBreak();
        _state->sfxBlockPlace = SoundEffect::generateBlockPlace();
        _state->sfxFootstep = SoundEffect::generateFootstep();
        _state->sfxWind = SoundEffect::generateAmbientWind();
        _state->sfxAnimal[0] = SoundEffect::generateSheepBaa();
        _state->sfxAnimal[1] = SoundEffect::generateCowMoo();
        _state->sfxAnimal[2] = SoundEffect::generatePigOink();
        _state->sfxAnimal[3] = SoundEffect::generateChickenCluck();
        [self syncAudioVolume];
    } else {
        RY_LOG_ERROR("Audio engine failed to initialize — continuing without sound");
        _state->audio.reset();
    }

    RY_LOG_INFO(std::string("Engine initialized — window: ") +
                std::to_string(static_cast<int>(_view.bounds.size.width)) + "x" +
                std::to_string(static_cast<int>(_view.bounds.size.height)) +
                ", device: " + std::string([[_device name] UTF8String]));

    return YES;
}

- (void)run {
    [_app run];
}

- (void)terminate {
    if (_app) {
        [_app terminate:nil];
    }
}

// ---- MTKViewDelegate: game loop with fixed timestep ----

- (void)drawInMTKView:(MTKView*)view {
    if (!_device || !_queue)
        return;

    EngineState* state = _state.get();

    // 1. Calculate elapsed time
    double currentTime = CACurrentMediaTime();
    double frameTime =
        (state->lastTime > 0) ? (currentTime - state->lastTime) : EngineState::TICK_DT;
    state->lastTime = currentTime;

    // Clamp frame time to prevent spiral of death after long pause
    if (frameTime > 0.25) {
        frameTime = 0.25;
    }

    state->deltaTime = frameTime;

    // 2. Add to accumulator
    state->accumulator += frameTime;

    // 3. Screen-level input (ESC, menu clicks, F3) — runs every frame so
    // menus stay responsive while the simulation is frozen
    [self handleGlobalInput];

    // 4. Fixed timestep game tick — menus freeze the world
    if (state->flow.screen == GameScreen::PLAYING) {
        while (state->accumulator >= EngineState::TICK_DT) {
            [self gameTick:state];
            state->accumulator -= EngineState::TICK_DT;
        }

        // Render the latest simulated position directly — no inter-tick
        // interpolation. Interpolation trails the sim by up to a tick (50 ms)
        // and read as floaty/light; the 20 Hz camera step is preferred over
        // that added latency. Falls stay gradual because velocity no longer
        // saturates (see Player::tick), so this no longer looks instantaneous.
        state->camera.setPosition(state->player.position + Vec3{0.f, Player::EYE_HEIGHT, 0.f});
    } else {
        state->accumulator = 0;
    }

    // 5. Render
    [self render];

    // 5. Increment frame count
    state->frameCount++;

    // 6. Consume input state for next frame
    if (state->inputManager) {
        state->inputManager->state().update();
    }

    (void)view;
}

- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)newSize {
    if (_renderPipeline && newSize.width > 0 && newSize.height > 0) {
        _renderPipeline->resize(static_cast<uint32_t>(newSize.width),
                                static_cast<uint32_t>(newSize.height));
    }
    (void)view;
}

// ---- Audio ----

// Master volume follows the settings slider while playing and mutes in
// menus (paused world = paused soundscape). The ambient wind bed starts on
// the first transition into gameplay.
- (void)syncAudioVolume {
    EngineState* state = _state.get();
    if (!state->audio)
        return;

    const bool playing = state->flow.screen == GameScreen::PLAYING;
    float volume = static_cast<float>(state->settings.volumeLevel) / 10.0f;
    state->audio->setMasterVolume(playing ? volume : 0.0f);

    if (playing && state->windVoice < 0 && !state->sfxWind.empty()) {
        state->windVoice = state->audio->playSound(state->sfxWind, SoundEffect::SAMPLE_RATE, 0.18f,
                                                   /*looping=*/true);
    }
}

- (void)playSfx:(const std::vector<float>&)buffer gain:(float)gain {
    EngineState* state = _state.get();
    if (!state->audio || state->flow.screen != GameScreen::PLAYING)
        return;
    state->audio->playSound(buffer, SoundEffect::SAMPLE_RATE, gain);
}

// ---- Screen-level input: ESC, menu interaction, debug HUD toggle ----

- (void)applyFlowEffects:(GameFlowEffects)effects {
    EngineState* state = _state.get();
    if (effects.captureCursor && state->inputManager) {
        state->inputManager->captureMouse();
    }
    if (effects.releaseCursor && state->inputManager) {
        state->inputManager->releaseMouse();
    }
    if (effects.resetTiming) {
        state->accumulator = 0;
        if (state->inputManager) {
            // Drop buffered look deltas AND pending tick presses — the click
            // that pressed RESUME must not break a block on the next tick
            state->inputManager->state().clearMouseDelta();
            state->inputManager->state().clearTickPresses();
        }
    }
    if (effects.requestQuit) {
        [self requestQuit];
    }
    [self syncAudioVolume];
}

// Settings steppers mutate live engine state; the screen doesn't change.
- (void)applySettingAction:(MenuAction)action {
    EngineState* state = _state.get();
    SettingsValues& settings = state->settings;

    switch (action) {
        case MenuAction::VIEW_DISTANCE_DOWN:
        case MenuAction::VIEW_DISTANCE_UP: {
            int step = (action == MenuAction::VIEW_DISTANCE_UP) ? 2 : -2;
            settings.viewDistance = std::clamp(settings.viewDistance + step, 4, 32);
            if (state->world) {
                state->world->setViewDistance(settings.viewDistance);
            }
            break;
        }
        case MenuAction::FOG_DOWN:
        case MenuAction::FOG_UP: {
            int step = (action == MenuAction::FOG_UP) ? 1 : -1;
            settings.fogLevel = std::clamp(settings.fogLevel + step, 0, 10);
            if (_renderPipeline) {
                _renderPipeline->setFogDensity(fogDensityForLevel(settings.fogLevel));
            }
            return;
        }
        case MenuAction::SENSITIVITY_DOWN:
        case MenuAction::SENSITIVITY_UP: {
            int step = (action == MenuAction::SENSITIVITY_UP) ? 1 : -1;
            settings.sensitivityLevel = std::clamp(settings.sensitivityLevel + step, 1, 10);
            state->camera.setMouseSensitivity(mouseSensitivityForLevel(settings.sensitivityLevel));
            return;
        }
        case MenuAction::VOLUME_DOWN:
        case MenuAction::VOLUME_UP: {
            int step = (action == MenuAction::VOLUME_UP) ? 1 : -1;
            settings.volumeLevel = std::clamp(settings.volumeLevel + step, 0, 10);
            [self syncAudioVolume];
            return;
        }
        case MenuAction::SHADOWS_DOWN:
        case MenuAction::SHADOWS_UP: {
            int step = (action == MenuAction::SHADOWS_UP) ? 1 : -1;
            state->gfx.shadowQuality = std::clamp(state->gfx.shadowQuality + step, 0,
                                                  GraphicsSettings::SHADOW_QUALITY_MAX);
            break;
        }
        case MenuAction::VL_TOGGLE:
            state->gfx.volumetricLight = !state->gfx.volumetricLight;
            break;
        case MenuAction::CLOUDS_DOWN:
        case MenuAction::CLOUDS_UP: {
            int step = (action == MenuAction::CLOUDS_UP) ? 1 : -1;
            state->gfx.cloudMode =
                std::clamp(state->gfx.cloudMode + step, 0, GraphicsSettings::CLOUD_MODE_MAX);
            break;
        }
        case MenuAction::SSAO_TOGGLE:
            state->gfx.ssao = !state->gfx.ssao;
            break;
        case MenuAction::SSR_TOGGLE:
            state->gfx.waterReflections = !state->gfx.waterReflections;
            break;
        case MenuAction::WAVING_TOGGLE:
            state->gfx.wavingFoliage = !state->gfx.wavingFoliage;
            break;
        case MenuAction::LENS_FLARE_TOGGLE:
            state->gfx.lensFlare = !state->gfx.lensFlare;
            break;
        case MenuAction::BLOOM_DOWN:
        case MenuAction::BLOOM_UP: {
            int step = (action == MenuAction::BLOOM_UP) ? 1 : -1;
            state->gfx.bloomLevel =
                std::clamp(state->gfx.bloomLevel + step, 0, GraphicsSettings::LEVEL_MAX);
            break;
        }
        case MenuAction::VIBRANCE_DOWN:
        case MenuAction::VIBRANCE_UP: {
            int step = (action == MenuAction::VIBRANCE_UP) ? 1 : -1;
            state->gfx.vibrance =
                std::clamp(state->gfx.vibrance + step, 0, GraphicsSettings::LEVEL_MAX);
            break;
        }
        case MenuAction::SHARPEN_DOWN:
        case MenuAction::SHARPEN_UP: {
            int step = (action == MenuAction::SHARPEN_UP) ? 1 : -1;
            state->gfx.sharpening =
                std::clamp(state->gfx.sharpening + step, 0, GraphicsSettings::LEVEL_MAX);
            break;
        }
        default:
            return; // not a settings action
    }

    // Only the video cases fall through: push the changed copy so the
    // renderer picks it up next frame.
    if (_renderPipeline) {
        _renderPipeline->setGraphicsSettings(state->gfx);
    }
}

- (void)handleGlobalInput {
    EngineState* state = _state.get();
    if (!state->inputManager)
        return;
    InputState& input = state->inputManager->state();

    if (input.isJustPressed(Key::Escape)) {
        // Leaving a settings screen persists the values (not per click —
        // no disk I/O while stepping)
        bool leavingSettings = state->flow.screen == GameScreen::SETTINGS ||
                               state->flow.screen == GameScreen::VIDEO_SETTINGS;
        [self applyFlowEffects:state->flow.onEscape()];
        if (leavingSettings && !state->envOverridesActive) {
            saveSettings(settingsPath(), state->settings, state->gfx);
        }
    }
    if (input.isJustPressed(Key::F3)) {
        state->showDebugHud = !state->showDebugHud;
    }

    if (state->flow.inMenu()) {
        // Rebuild the layout each frame (settings values can change) and
        // hit-test in window points — the same normalized space it's drawn in
        float boundsW = static_cast<float>(_view.bounds.size.width);
        float boundsH = static_cast<float>(_view.bounds.size.height);
        state->menuLayout =
            buildMenuLayout(state->flow.screen, boundsW, boundsH, state->settings, state->gfx);

        Vec2 mouse = input.mousePosition;
        state->hoveredButton = menuHitTest(state->menuLayout, mouse.x / boundsW, mouse.y / boundsH);

        if (input.isJustPressed(Key::MouseLeft) && state->hoveredButton >= 0) {
            MenuAction action =
                state->menuLayout.buttons[static_cast<size_t>(state->hoveredButton)].action;
            [self applySettingAction:action];
            [self applyFlowEffects:state->flow.onMenuAction(action)];
            if ((action == MenuAction::CLOSE_SETTINGS ||
                 action == MenuAction::CLOSE_VIDEO_SETTINGS) &&
                !state->envOverridesActive) {
                saveSettings(settingsPath(), state->settings, state->gfx);
            }
        }
    } else {
        state->hoveredButton = -1;

        // Hotbar: scroll wheel cycles slots (frame-level, gameplay only)
        _scrollAccumulator += input.scrollDelta;
        while (_scrollAccumulator >= 10.0f) {
            state->hotbar.selectNext();
            _scrollAccumulator -= 10.0f;
        }
        while (_scrollAccumulator <= -10.0f) {
            state->hotbar.selectPrev();
            _scrollAccumulator += 10.0f;
        }
    }
}

// ---- Clean quit: save the world, then terminate through AppKit ----

- (void)saveWorldState {
    if (_savedWorld)
        return;
    // Capture runs are throwaway playtests: their spawned test blocks
    // (RYCRAFT_SPAWN_*) and drifted player position must not overwrite the
    // real save, and reproducible captures depend on the world not moving.
    if (std::getenv("RYCRAFT_CAPTURE")) {
        _savedWorld = true;
        return;
    }
    _savedWorld = true;

    // Mesh workers reference the World — stop them before anything else
    // (ivar destruction order at teardown is not something to bet on)
    if (_renderPipeline) {
        _renderPipeline->shutdownMeshWorkers();
    }

    EngineState* state = _state.get();
    if (state->saveManager && state->world) {
        // Edited chunks persist on unload; the quit path sweeps the rest
        state->world->saveModifiedChunks();
        state->saveManager->saveMetadata(state->world->getSeed(), state->player.position,
                                         state->worldTime);
        state->saveManager->flush();
        RY_LOG_INFO("World state saved");
    }

    // Settings share the quit path so mid-session tweaks survive a close
    // that never revisited the settings screen. Env-overridden playtest
    // sessions never save — their overrides must not become the file.
    if (!state->envOverridesActive) {
        saveSettings(settingsPath(), state->settings, state->gfx);
    }
}

- (void)requestQuit {
    if (_state->inputManager) {
        _state->inputManager->releaseMouse();
    }
    [NSApp terminate:self];
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication*)sender {
    (void)sender;
    [self saveWorldState];
    return NSTerminateNow;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
    (void)sender;
    // The red close button routes through applicationShouldTerminate: above,
    // so the save-on-quit path has no duplicate
    return YES;
}

- (void)applicationWillTerminate:(NSNotification*)notification {
    (void)notification;
    // Belt and braces: the mouse association is global system state
    CGAssociateMouseAndMouseCursorPosition(true);
    [self saveWorldState];
}

- (void)windowDidResignKey:(NSNotification*)notification {
    (void)notification;
    // Cmd-Tab (or any focus loss) must never leave the pointer locked
    [self applyFlowEffects:_state->flow.onFocusLost()];
}

// ---- Game Tick (fixed timestep at 20Hz) ----

- (void)gameTick:(EngineState*)state {
    if (!state->inputManager || !state->world)
        return;

    InputState& input = state->inputManager->state();

    // Playtest hook: RYCRAFT_AUTOPILOT=walk holds W down so headless runs
    // can verify the full input→tick→physics path end to end; =sprint also
    // holds the sprint key so captures can show the FOV widening
    static const char* autopilot = std::getenv("RYCRAFT_AUTOPILOT");
    if (autopilot) {
        const std::string_view mode{autopilot};
        if (mode == "walk" || mode == "sprint") {
            input.keysDown[Key::W] = true;
        }
        if (mode == "sprint") {
            input.keysDown[Key::LeftControl] = true;
        }
    }

    // 1. Advance world time (1 tick per game tick). Playtest hooks:
    // RYCRAFT_TIME=<0..23999> pins the time of day at launch and
    // RYCRAFT_TIME_FREEZE=1 stops it advancing, so captures at noon /
    // sunset / midnight don't depend on hand-editing the save.
    static const bool freezeTime = [] {
        const char* f = std::getenv("RYCRAFT_TIME_FREEZE");
        return f && *f && std::strcmp(f, "0") != 0;
    }();
    if (!freezeTime) {
        state->worldTime++;
    }

    // 1b. Unstick a stale spawn: a resumed save can place the player inside
    // terrain when world generation has changed shape since the save was
    // written — collision then zeroes every move. Once the spawn chunk
    // exists, lift the player to the surface if they are embedded.
    if (!state->spawnValidated) {
        int px = static_cast<int>(std::floor(state->player.position.x));
        int pz = static_cast<int>(std::floor(state->player.position.z));
        auto chunk = state->world->getChunk(Chunk::worldToChunk(px), Chunk::worldToChunk(pz));
        if (chunk && chunk->generated) {
            int feetY = static_cast<int>(std::floor(state->player.position.y));
            bool embedded = isSolid(state->world->getBlock(px, feetY, pz)) ||
                            isSolid(state->world->getBlock(px, feetY + 1, pz));
            if (embedded) {
                for (int y = CHUNK_HEIGHT - 2; y > 0; --y) {
                    if (isSolid(state->world->getBlock(px, y, pz))) {
                        state->player.position.y = static_cast<float>(y + 1);
                        state->player.velocity = Vec3{0.f, 0.f, 0.f};
                        RY_LOG_INFO("Spawn was inside terrain — moved player to the surface");
                        break;
                    }
                }
            }

            // Playtest hook: RYCRAFT_SPAWN_LAVA=1 carves a small surface lava
            // pool a few blocks from spawn so headless captures can show the
            // block-light glow (natural lava only forms in deep caves). Best
            // paired with a night RYCRAFT_TIME so the orange spill stands out.
            static const bool spawnLava = [] {
                const char* v = std::getenv("RYCRAFT_SPAWN_LAVA");
                return v && *v && std::strcmp(v, "0") != 0;
            }();
            if (spawnLava) {
                for (int dz = -2; dz <= 2; ++dz) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        int wx = px + dx, wz = pz + 6 + dz; // ahead (+Z) at spawn
                        for (int y = CHUNK_HEIGHT - 2; y > 0; --y) {
                            if (isSolid(state->world->getBlock(wx, y, wz))) {
                                state->world->setBlock(wx, y, wz, BlockType::LAVA);
                                state->world->setBlock(wx, y + 1, wz, BlockType::AIR);
                                break;
                            }
                        }
                    }
                }
            }

            // Playtest hook: RYCRAFT_SPAWN_WATER=1 sinks a broad pool ahead of
            // spawn (between the camera and the tree line) so headless captures
            // can show the water reflections/refraction at a grazing angle.
            static const bool spawnWater = [] {
                const char* v = std::getenv("RYCRAFT_SPAWN_WATER");
                return v && *v && std::strcmp(v, "0") != 0;
            }();
            if (spawnWater) {
                auto isTrunk = [](BlockType b) {
                    return b == BlockType::LOG || b == BlockType::BIRCH_LOG ||
                           b == BlockType::SPRUCE_LOG;
                };
                for (int dz = 3; dz <= 22; ++dz) {
                    for (int dx = -10; dx <= 10; ++dx) {
                        int wx = px + dx, wz = pz + dz;
                        // Scan down to the terrain surface, past any tree trunk,
                        // so the pool sits at ground level (no floating water).
                        for (int y = CHUNK_HEIGHT - 2; y > 0; --y) {
                            BlockType b = state->world->getBlock(wx, y, wz);
                            if (isSolid(b) && !isTrunk(b)) {
                                state->world->setBlock(wx, y, wz, BlockType::WATER);
                                state->world->setBlock(wx, y - 1, wz, BlockType::WATER);
                                state->world->setBlock(wx, y + 1, wz, BlockType::AIR);
                                break;
                            }
                        }
                    }
                }
            }

            // Playtest hook: RYCRAFT_YAW / RYCRAFT_PITCH (degrees) point the
            // camera for captures — e.g. face the afternoon sun for the lens
            // flare. Yaw 0 looks +Z; -90 looks -X.
            static const char* yawEnv = std::getenv("RYCRAFT_YAW");
            static const char* pitchEnv = std::getenv("RYCRAFT_PITCH");
            if (yawEnv || pitchEnv) {
                constexpr float DEG = static_cast<float>(M_PI) / 180.0f;
                state->camera.setLook(yawEnv ? std::atof(yawEnv) * DEG : 0.0f,
                                      pitchEnv ? std::atof(pitchEnv) * DEG : 0.0f);
            }

            state->spawnValidated = true;
        }
    }

    // 2. Hotbar input: keys 1-9 select slots
    for (int i = 0; i < Hotbar::SLOTS; ++i) {
        Key key = static_cast<Key>(static_cast<int>(Key::One) + i);
        if (input.isPressedForTick(key)) {
            state->hotbar.selectSlot(i);
            break;
        }
    }

    // 3. Update camera from player state (at EYE height, not the feet), then
    // consume the look delta so a second tick in the same frame doesn't
    // re-apply it
    Vec3 eyePosition = state->player.position + Vec3{0.f, Player::EYE_HEIGHT, 0.f};
    state->camera.setPosition(eyePosition);
    InputBindings bindings;
    state->camera.update(state->deltaTime, input, bindings, eyePosition);
    input.clearMouseDelta();

    // 4. Sync player yaw/pitch from camera so WASD uses the correct
    // direction and swimming can follow the look direction
    state->player.yaw = state->camera.yaw();
    state->player.pitch = state->camera.pitch();

    // 5. Player physics tick. All movement keys are decoded through the
    // bindings here (sprint once read a hardcoded key and fired on sneak);
    // Player itself never touches the input layer.
    PlayerInput playerInput;
    playerInput.forward = input.isDown(bindings.forward.key);
    playerInput.backward = input.isDown(bindings.backward.key);
    playerInput.left = input.isDown(bindings.left.key);
    playerInput.right = input.isDown(bindings.right.key);
    playerInput.jumpHeld = input.isDown(bindings.jump.key);
    playerInput.jumpPressed = input.isPressedForTick(bindings.jump.key);
    playerInput.sprintHeld = input.isDown(bindings.sprint.key);
    playerInput.descendHeld = input.isDown(bindings.sneak.key);
    playerInput.doubleTapForward = input.isDoubleTappedForTick(bindings.forward.key);
    playerInput.doubleTapJump = input.isDoubleTappedForTick(bindings.jump.key);
    state->player.tick(*state->world, playerInput);

    // 5b. Footsteps: one thud roughly every two blocks walked on the ground
    if (state->audio) {
        Vec3 moved = state->player.position - state->lastFootstepPos;
        moved.y = 0.f;
        state->lastFootstepPos = state->player.position;
        if (state->player.onGround) {
            state->footstepDistance += moved.length();
            if (state->footstepDistance >= 2.2f) {
                state->footstepDistance = 0.f;
                [self playSfx:state->sfxFootstep gain:0.5f];
            }
        } else {
            state->footstepDistance = 0.f;
        }
    }

    // 6. Update loaded chunks around player
    // updatePlayerPosition takes WORLD coordinates (it converts to chunk
    // coords itself) — passing pre-converted chunk coords made streaming
    // track chunk/16, so the world unloaded under the player ~200 blocks out
    state->world->updatePlayerPosition(static_cast<int>(std::floor(state->player.position.x)),
                                       static_cast<int>(std::floor(state->player.position.z)));

    // 6b. Animals: initial population once the spawn area streamed in, then
    // per-tick AI steering + physics for every living entity
    if (state->spawner) {
        if (!state->populationSpawned && state->world->getPendingChunkCount() == 0) {
            // Populate only the chunks near spawn, with a hard cap — biome
            // densities over the full view distance produce thousands of
            // animals, which neither the AI tick nor the player needs.
            constexpr int SPAWN_CHUNK_RADIUS = 3;
            constexpr size_t MAX_ANIMALS = 64;
            int playerChunkX = Chunk::worldToChunk(static_cast<int>(state->player.position.x));
            int playerChunkZ = Chunk::worldToChunk(static_cast<int>(state->player.position.z));
            for (int dz = -SPAWN_CHUNK_RADIUS; dz <= SPAWN_CHUNK_RADIUS; ++dz) {
                for (int dx = -SPAWN_CHUNK_RADIUS; dx <= SPAWN_CHUNK_RADIUS; ++dx) {
                    if (state->spawner->getEntities().size() >= MAX_ANIMALS)
                        break;
                    state->spawner->spawnForChunk(playerChunkX + dx, playerChunkZ + dz);
                }
            }
            state->populationSpawned = true;
            RY_LOG_INFO(std::string("Spawned ") +
                        std::to_string(state->spawner->getEntities().size()) + " animals");
        }

        auto& entities = state->spawner->getEntities();
        auto& spatialHash = state->spawner->getSpatialHash();
        // Index-based over a size snapshot with a shared_ptr copy: breeding
        // inside StateMachine::update can push_back into this very vector,
        // which would invalidate range-for iterators mid-loop.
        const size_t entityCount = entities.size();
        for (size_t i = 0; i < entityCount; ++i) {
            std::shared_ptr<Entity> entity = entities[i];
            if (!entity || !entity->alive)
                continue;

            // Simulation distance: distant animals stand still (cheap and
            // invisible — they are beyond the fog anyway)
            Vec3 offset = entity->position - state->player.position;
            if (offset.length() > 96.f)
                continue;

            StateMachine& brain = state->entityBrains[entity->id];
            Vec3 steering = brain.update(*entity, *state->world, state->player.position,
                                         /*playerMovingToward=*/false,
                                         /*playerHoldingFood=*/false, *state->spawner);
            entity->velocity.x += steering.x;
            entity->velocity.z += steering.z;
            entity->tick(*state->world);

            // Keep the spatial hash in sync for flocking neighbor queries
            spatialHash.remove(entity->id);
            spatialHash.insert(entity->id, entity->position);
        }

        // Occasional ambient animal call from something nearby
        if (state->animalCallCooldown > 0) {
            --state->animalCallCooldown;
        } else if (state->audio && !entities.empty()) {
            static SeededRng callRng(0xA111CA11u);
            const auto& candidate =
                entities[callRng.nextInt(0, static_cast<int>(entities.size()) - 1)];
            if (candidate && candidate->alive) {
                Vec3 toPlayer = candidate->position - state->player.position;
                if (toPlayer.length() < 24.f && callRng.nextFloat() < 0.3f) {
                    int type = static_cast<int>(candidate->type);
                    [self playSfx:state->sfxAnimal[type] gain:0.35f];
                }
            }
            state->animalCallCooldown = 120 + callRng.nextInt(0, 120); // 6-12s
        }
    }

    // 7. Weather: a deterministic schedule from the world seed (convention:
    // seeded randomness only) — each in-game day hashes to a maybe-rain
    // window, so playtests reproduce exactly. Playtest hook:
    // RYCRAFT_WEATHER=rain|clear pins the state.
    {
        // hash64 is the engine's one seeded-hash home (common/random.hpp)
        uint64_t day = state->worldTime / EngineState::TICKS_PER_DAY;
        uint64_t h = hash64(day ^ (static_cast<uint64_t>(state->world->getSeed()) << 32));
        uint32_t tod = static_cast<uint32_t>(state->worldTime % EngineState::TICKS_PER_DAY);
        uint32_t start = 3000u + static_cast<uint32_t>((h >> 8) % 12000u);
        uint32_t length = 2000u + static_cast<uint32_t>((h >> 4) % 4000u);
        state->raining =
            (h % 100u) < EngineState::RAIN_DAYS_PERCENT && tod >= start && tod < start + length;
        static const char* weatherEnv = std::getenv("RYCRAFT_WEATHER");
        if (weatherEnv) {
            // Playtest override skips the soak ramp so captures are stable.
            state->raining = std::strcmp(weatherEnv, "rain") == 0;
            state->wetness = state->raining ? 1.0f : 0.0f;
        } else {
            // Fixed-step tick logic uses the tick dt, not the frame delta.
            const float dt = static_cast<float>(EngineState::TICK_DT);
            state->wetness = state->raining
                                 ? std::min(1.0f, state->wetness + dt / EngineState::SOAK_SECONDS)
                                 : std::max(0.0f, state->wetness - dt / EngineState::DRY_SECONDS);
        }
    }
    if (_renderPipeline) {
        // Particles integrate per fixed tick too — the frame delta here made
        // rainfall speed depend on the frame rate (1/3 speed at 60 FPS).
        _renderPipeline->tickParticles(static_cast<float>(EngineState::TICK_DT), *state->world,
                                       state->player.position, state->raining);
        _renderPipeline->setWetness(state->wetness);
    }

    // 8. Single raycast for block interaction + highlight
    Vec3 cameraPos = state->camera.position();
    Vec3 forward = state->camera.forward();
    auto rayHit = VoxelTraversal::traceRayWithNormal(cameraPos, forward, *state->world, 6.0f);

    // Update block highlight
    if (rayHit.has_value()) {
        state->highlightedBlock = rayHit->first;
        state->hasHighlightedBlock = true;
    } else {
        state->hasHighlightedBlock = false;
    }

    // Block breaking (left mouse click)
    if (input.isPressedForTick(Key::MouseLeft)) {
        [self breakBlock:state hit:rayHit];
    }

    // Block placing (right mouse click)
    if (input.isPressedForTick(Key::MouseRight)) {
        [self placeBlock:state hit:rayHit];
    }

    // Tick-edge input consumed — a second tick in this frame must not re-fire
    input.clearTickPresses();
}

// ---- Block Breaking (Task 6.1) ----

- (void)breakBlock:(EngineState*)state hit:(BlockRayHit)hit {
    if (!hit.has_value())
        return;

    int hitX = static_cast<int>(std::floor(hit->first.x));
    int hitY = static_cast<int>(std::floor(hit->first.y));
    int hitZ = static_cast<int>(std::floor(hit->first.z));

    // Cannot break bedrock
    BlockType current = state->world->getBlock(hitX, hitY, hitZ);
    if (current == BlockType::BEDROCK)
        return;

    // Set block to air
    state->world->setBlock(hitX, hitY, hitZ, BlockType::AIR);

    // Flora standing on the broken block loses its support and pops with it
    // (same column → same chunk → the dirty/save below covers it)
    if (isFlora(state->world->getBlock(hitX, hitY + 1, hitZ))) {
        state->world->setBlock(hitX, hitY + 1, hitZ, BlockType::AIR);
    }

    // World::setBlock marks the chunk (and boundary neighbors) dirty and
    // flags it for save-on-unload

    // Clear highlight since block is now air
    state->hasHighlightedBlock = false;

    [self playSfx:state->sfxBlockBreak gain:0.8f];
}

// ---- Block Placing (Task 6.2) ----

- (void)placeBlock:(EngineState*)state hit:(BlockRayHit)hit {
    if (!hit.has_value())
        return;

    // Calculate placement position: hit block + face normal
    int placeX = static_cast<int>(std::floor(hit->first.x)) + static_cast<int>(hit->second.x);
    int placeY = static_cast<int>(std::floor(hit->first.y)) + static_cast<int>(hit->second.y);
    int placeZ = static_cast<int>(std::floor(hit->first.z)) + static_cast<int>(hit->second.z);

    // Validate: placement AABB must not overlap player AABB
    AABB placeBox{
        Vec3{static_cast<float>(placeX), static_cast<float>(placeY), static_cast<float>(placeZ)},
        Vec3{static_cast<float>(placeX + 1), static_cast<float>(placeY + 1),
             static_cast<float>(placeZ + 1)}};

    if (placeBox.intersects(state->player.getAABB()))
        return;

    // Place block (World::setBlock marks the chunk and boundary neighbors
    // dirty and flags the chunk for save-on-unload)
    BlockType selectedType = state->hotbar.getSelectedBlockType();
    state->world->setBlock(placeX, placeY, placeZ, selectedType);

    [self playSfx:state->sfxBlockPlace gain:0.8f];
}

// ---- Block Highlight Update (Task 6.9) ----
// Merged into gameTick: via single raycast (Major #4 fix)

// ---- Render ----

- (void)render {
    EngineState* state = _state.get();

    // Ease the FOV toward the movement mode's target (dt-correct exponential,
    // so the zoom speed doesn't depend on frame rate) and hand it to the
    // camera — the cloud shader reads camera.FOV(), so routing the projection
    // through the same value keeps clouds registered with the world mid-zoom.
    float targetFov = state->player.sprinting ? EngineState::SPRINT_FOV : EngineState::BASE_FOV;
    state->fovCurrent +=
        (targetFov - state->fovCurrent) *
        (1.0f - std::exp(-static_cast<float>(state->deltaTime) / EngineState::FOV_EASE_SECONDS));
    state->camera.setFOV(state->fovCurrent);

    // Update projection matrix from current drawable size
    CGSize currentSize = _view.drawableSize;
    if (currentSize.width > 0 && currentSize.height > 0) {
        state->drawableSize = currentSize;
        float aspect =
            static_cast<float>(currentSize.width) / static_cast<float>(currentSize.height);
        state->projectionMatrix = Mat4::perspective(
            state->camera.FOV() * (static_cast<float>(M_PI) / 180.0f), aspect, 0.1f, 1000.0f);
    }

    id<CAMetalDrawable> drawable = _view.currentDrawable;
    if (!drawable)
        return;

    if (!_renderPipeline || !state->world)
        return;

    // Log render + streaming diagnostics every 60 frames (the same numbers
    // the F3 HUD shows, so headless playtests can measure against budgets;
    // the chunk count reuses the HUD's 30-frame sample instead of copying
    // the whole chunk vector)
    if (state->frameCount % 60 == 1) {
        auto chunkStats = _renderPipeline->chunkRenderStats();
        char line[288];
        snprintf(line, sizeof(line),
                 "Render: %u loaded chunks, frame %llu player (%.1f, %.1f, %.1f) | %.2f ms/frame "
                 "gpu %.2f ms gen %.2f ms mesh %.2f ms pending %zu vram %.0f/%.0f MB",
                 state->cachedChunkCount, static_cast<unsigned long long>(state->frameCount),
                 state->player.position.x, state->player.position.y, state->player.position.z,
                 state->smoothedFrameMs, _renderPipeline->gpuFrameMs(),
                 state->world->averageGenMs(), chunkStats.meshMsAvg,
                 state->world->getPendingChunkCount(), chunkStats.megaUsedMB, chunkStats.megaCapMB);
        RY_LOG_INFO(line);
        // Per-pass GPU breakdown (RYCRAFT_GPU_COUNTERS=1) mirrors to the log
        // so headless runs can attribute frame cost to individual passes.
        std::string passes = _renderPipeline->gpuPassBreakdown();
        if (!passes.empty()) {
            RY_LOG_INFO(("GPU passes (ms): " + passes).c_str());
        }
    }

    // Playtest hook: RYCRAFT_CAPTURE=<path.png> writes one frame to disk
    // once RYCRAFT_CAPTURE_FRAME (default 240) frames have rendered, then
    // quits ~1s later (the PNG write is async). A capture run is headless
    // tooling — leaving it running leaked a full game instance per capture
    // until concurrent playtests exhausted system memory.
    static const char* capturePath = std::getenv("RYCRAFT_CAPTURE");
    if (capturePath && *capturePath) {
        static const uint64_t captureFrame = [] {
            const char* frameEnv = std::getenv("RYCRAFT_CAPTURE_FRAME");
            return frameEnv ? static_cast<uint64_t>(std::atoll(frameEnv)) : uint64_t{240};
        }();
        if (state->frameCount == captureFrame) {
            _renderPipeline->requestFrameCapture(capturePath);
        }
        if (state->frameCount == captureFrame + 60) {
            [self requestQuit];
        }
    }

    // Camera view matrix
    Mat4 viewMatrix = state->camera.viewMatrix();

    // Real performance stats for the F3 HUD (EMA-smoothed frame time; chunk
    // count sampled every 30 frames — getLoadedChunks copies under a lock)
    state->smoothedFrameMs =
        state->smoothedFrameMs * 0.95f + static_cast<float>(state->deltaTime) * 1000.0f * 0.05f;
    if (state->frameCount % 30 == 0) {
        state->cachedChunkCount = static_cast<uint32_t>(state->world->getLoadedChunks().size());
        state->cachedPendingChunks = static_cast<uint32_t>(state->world->getPendingChunkCount());
    }

    UIFrameState uiFrame;
    uiFrame.screen = state->flow.screen;
    uiFrame.hoveredButton = state->hoveredButton;
    uiFrame.showDebugHud = state->showDebugHud;
    // Underwater view (veil, god rays, dense fog): the camera cell is water.
    // Non-generating read — a streaming lag must never stall the frame.
    {
        Vec3 camPos = state->camera.getPosition();
        uiFrame.cameraUnderwater =
            state->world->getBlockIfLoaded(
                static_cast<int>(std::floor(camPos.x)), static_cast<int>(std::floor(camPos.y)),
                static_cast<int>(std::floor(camPos.z))) == BlockType::WATER;
    }
    uiFrame.stats.frameTimeMs = state->smoothedFrameMs;
    uiFrame.stats.gpuFrameMs = _renderPipeline->gpuFrameMs();
    uiFrame.stats.fps = state->smoothedFrameMs > 0.f ? 1000.0f / state->smoothedFrameMs : 0.f;
    uiFrame.stats.chunkCount = state->cachedChunkCount;
    uiFrame.stats.entityCount =
        state->spawner ? static_cast<uint32_t>(state->spawner->getEntities().size()) : 0;
    uiFrame.stats.pendingChunks = state->cachedPendingChunks;
    uiFrame.stats.genMsAvg = state->world->averageGenMs();
    auto chunkStats = _renderPipeline->chunkRenderStats();
    uiFrame.stats.meshMsAvg = chunkStats.meshMsAvg;
    uiFrame.stats.meshBuildsFrame = chunkStats.meshBuildsLastFrame;
    uiFrame.stats.megaUsedMB = chunkStats.megaUsedMB;
    uiFrame.stats.megaCapMB = chunkStats.megaCapMB;
    uiFrame.menu = state->menuLayout;

    _renderPipeline->render(
        _queue, drawable, viewMatrix, state->projectionMatrix, *state->world, state->camera,
        state->worldTime,
        state->hasHighlightedBlock ? std::optional<Vec3>(state->highlightedBlock) : std::nullopt,
        state->hotbar, uiFrame, state->spawner ? &state->spawner->getEntities() : nullptr);
}

- (double)deltaTime {
    return _state->deltaTime;
}

- (uint64_t)frameCount {
    return _state->frameCount;
}

@end

// ---- C++ bridge functions (visible from header) ----

Mat4 engineProjectionMatrix(Engine* engine) {
    // Access internal state via the static helper defined in @implementation
    extern EngineState* _engineGetState(Engine*);
    EngineState* state = _engineGetState(engine);
    return state ? state->projectionMatrix : Mat4::identity();
}

CGSize engineDrawableSize(Engine* engine) {
    extern EngineState* _engineGetState(Engine*);
    EngineState* state = _engineGetState(engine);
    if (!state)
        return {0, 0};
    return state->drawableSize;
}

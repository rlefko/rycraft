#import <engine/engine.hpp>

#import <QuartzCore/QuartzCore.h>

#include <common/error.hpp>
#include <common/math.hpp>
#include <render/render_pipeline.hpp>
#include <engine/camera.hpp>
#include <engine/game_state.hpp>
#include <engine/hotbar.hpp>
#include <engine/input_bindings.hpp>
#include <render/ui_menu.hpp>
#include <entity/player.hpp>
#include <entity/voxel_traversal.hpp>
#include <world/world.hpp>
#include <world/save_manager.hpp>
#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>
#include <optional>
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

    // ---- Day/Night Cycle ----
    static constexpr uint64_t TICKS_PER_DAY = 24000; // 20 min at 20Hz
    uint64_t worldTime = 0;

    // ---- Block Interaction ----
    Hotbar hotbar;
    Vec3 highlightedBlock; // Block currently targeted by crosshair
    bool hasHighlightedBlock = false;

    // ---- Game flow & UI ----
    GameFlow flow;                 // Title → Playing ⇄ Paused ⇄ Settings
    SettingsValues settings;       // live values shown in the settings menu
    MenuLayout menuLayout;         // rebuilt each frame while a menu is open
    int hoveredButton = -1;
    bool showDebugHud = false;

    // ---- Performance stats (exponential moving averages) ----
    float smoothedFrameMs = 16.7f;
    uint32_t cachedChunkCount = 0;

    // ---- Player & World ----
    Player player;
    std::shared_ptr<World> world;
    std::unique_ptr<SaveManager> saveManager;
    Camera camera;

    // ---- Audio ----
    std::unique_ptr<AudioEngine> audio;
    std::vector<float> sfxBlockBreak;
    std::vector<float> sfxBlockPlace;
    std::vector<float> sfxFootstep;
    std::vector<float> sfxWind;
    int32_t windVoice = -1;
    float footstepDistance = 0.f;   // ground distance walked since last step
    Vec3 lastFootstepPos{0.f, 0.f, 0.f};

    // ---- Input manager (set after window creation) ----
    InputManager* inputManager = nullptr;
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

        // View distance 12 keeps the full-detail mesh set comfortably inside
        // the 128 MB mega-buffer (25×25 chunks ≈ 60 MB of vertex data).
        _state->world = std::make_shared<World>(seed, _state->settings.viewDistance);
        // Chunks load from disk before regenerating, so block edits persist
        _state->world->setSaveManager(_state->saveManager.get());
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
    [_app setActivationPolicy: NSApplicationActivationPolicyRegular];
    [_app activateIgnoringOtherApps: true];
    [_app setDelegate: self];

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
    NSRect windowRect = NSMakeRect(
        (screenFrame.size.width - 1024) * 0.5,
        (screenFrame.size.height - 768) * 0.5,
        1024, 768
    );
    _window = [[NSWindow alloc]
        initWithContentRect: windowRect
        styleMask: NSWindowStyleMaskTitled
             | NSWindowStyleMaskClosable
             | NSWindowStyleMaskMiniaturizable
             | NSWindowStyleMaskResizable
        backing: NSBackingStoreBuffered
        defer: false];
    if (!_window) {
        RY_LOG_FATAL("Failed to create NSWindow");
        return NO;
    }
    [_window setTitle: @"rycraft"];
    [_window setDelegate: self];
    [_window makeKeyAndOrderFront: nil];

    // 5. Create and configure MTKView
    _view = [[MTKView alloc]
        initWithFrame: windowRect
        device: _device];
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
    [_window setContentView: _view];

    // 7. Create InputManager (the game opens on the title screen with a
    // free cursor; clicking PLAY captures the mouse)
    _state->inputManager = new InputManager(_window);

    // 6. Load shader library and create render pipeline
    NSString* exePath = [[NSBundle mainBundle] executablePath];
    NSString* dirPath = [exePath stringByDeletingLastPathComponent];
    NSString* libPath = [dirPath stringByAppendingPathComponent:@"pipeline.metallib"];
    NSURL* libURL = [NSURL fileURLWithPath:libPath];
    NSError* libError = nil;
    id<MTLLibrary> library = [_device newLibraryWithURL:libURL
                                                    error:&libError];
    if (libError) {
        NSString* msg = [NSString stringWithFormat:@"Failed to load shader library: %@",
                         libError.localizedDescription];
        RY_LOG_FATAL([msg UTF8String]);
    }

    _renderPipeline = std::make_unique<RenderPipeline>(
        _device,
        library,
        static_cast<uint32_t>(_view.bounds.size.width),
        static_cast<uint32_t>(_view.bounds.size.height)
    );

    // Playtest/diagnostic override: RYCRAFT_BLOOM=<0..1> scales or disables bloom
    if (const char* bloomEnv = std::getenv("RYCRAFT_BLOOM")) {
        _renderPipeline->setBloomIntensity(static_cast<float>(std::atof(bloomEnv)));
    }

    // Playtest override: start on a specific screen (title|playing|paused|settings)
    if (const char* screenEnv = std::getenv("RYCRAFT_START_SCREEN")) {
        std::string name = screenEnv;
        if (name == "playing") {
            _state->flow.screen = GameScreen::Playing;
            _state->inputManager->captureMouse();
        } else if (name == "paused") {
            _state->flow.screen = GameScreen::Paused;
        } else if (name == "settings") {
            _state->flow.screen = GameScreen::Settings;
        }
    }

    // 8. Audio: non-fatal on failure — the game is fully playable silent
    _state->audio = std::make_unique<AudioEngine>();
    if (_state->audio->initialize()) {
        _state->sfxBlockBreak = SoundEffect::generateBlockBreak();
        _state->sfxBlockPlace = SoundEffect::generateBlockPlace();
        _state->sfxFootstep = SoundEffect::generateFootstep();
        _state->sfxWind = SoundEffect::generateAmbientWind();
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
        [_app terminate: nil];
    }
}

// ---- MTKViewDelegate: game loop with fixed timestep ----

- (void)drawInMTKView:(MTKView*)view {
    if (!_device || !_queue) return;

    EngineState* state = _state.get();

    // 1. Calculate elapsed time
    double currentTime = CACurrentMediaTime();
    double frameTime = (state->lastTime > 0) ? (currentTime - state->lastTime) : EngineState::TICK_DT;
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
    if (state->flow.screen == GameScreen::Playing) {
        while (state->accumulator >= EngineState::TICK_DT) {
            [self gameTick:state];
            state->accumulator -= EngineState::TICK_DT;
        }
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
        _renderPipeline->resize(
            static_cast<uint32_t>(newSize.width),
            static_cast<uint32_t>(newSize.height)
        );
    }
    (void)view;
}

// ---- Audio ----

// Master volume follows the settings slider while playing and mutes in
// menus (paused world = paused soundscape). The ambient wind bed starts on
// the first transition into gameplay.
- (void)syncAudioVolume {
    EngineState* state = _state.get();
    if (!state->audio) return;

    const bool playing = state->flow.screen == GameScreen::Playing;
    float volume = static_cast<float>(state->settings.volumeLevel) / 10.0f;
    state->audio->setMasterVolume(playing ? volume : 0.0f);

    if (playing && state->windVoice < 0 && !state->sfxWind.empty()) {
        state->windVoice = state->audio->playSound(state->sfxWind, SoundEffect::SAMPLE_RATE,
                                                   0.18f, /*looping=*/true);
    }
}

- (void)playSfx:(const std::vector<float>&)buffer gain:(float)gain {
    EngineState* state = _state.get();
    if (!state->audio || state->flow.screen != GameScreen::Playing) return;
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
            state->inputManager->state().clearMouseDelta();
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
        case MenuAction::ViewDistanceDown:
        case MenuAction::ViewDistanceUp: {
            int step = (action == MenuAction::ViewDistanceUp) ? 2 : -2;
            settings.viewDistance = std::clamp(settings.viewDistance + step, 4, 32);
            if (state->world) {
                state->world->setViewDistance(settings.viewDistance);
            }
            break;
        }
        case MenuAction::FogDown:
        case MenuAction::FogUp: {
            int step = (action == MenuAction::FogUp) ? 1 : -1;
            settings.fogLevel = std::clamp(settings.fogLevel + step, 0, 10);
            if (_renderPipeline) {
                _renderPipeline->setFogDensity(static_cast<float>(settings.fogLevel) * 0.0001f);
            }
            break;
        }
        case MenuAction::SensitivityDown:
        case MenuAction::SensitivityUp: {
            int step = (action == MenuAction::SensitivityUp) ? 1 : -1;
            settings.sensitivityLevel = std::clamp(settings.sensitivityLevel + step, 1, 10);
            state->camera.setMouseSensitivity(
                static_cast<float>(settings.sensitivityLevel) * 0.0005f);
            break;
        }
        case MenuAction::VolumeDown:
        case MenuAction::VolumeUp: {
            int step = (action == MenuAction::VolumeUp) ? 1 : -1;
            settings.volumeLevel = std::clamp(settings.volumeLevel + step, 0, 10);
            [self syncAudioVolume];
            break;
        }
        default:
            break;
    }
}

- (void)handleGlobalInput {
    EngineState* state = _state.get();
    if (!state->inputManager) return;
    InputState& input = state->inputManager->state();

    if (input.isJustPressed(Key::Escape)) {
        [self applyFlowEffects:state->flow.onEscape()];
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
            buildMenuLayout(state->flow.screen, boundsW, boundsH, state->settings);

        Vec2 mouse = input.mousePosition;
        state->hoveredButton =
            menuHitTest(state->menuLayout, mouse.x / boundsW, mouse.y / boundsH);

        if (input.isJustPressed(Key::MouseLeft) && state->hoveredButton >= 0) {
            MenuAction action =
                state->menuLayout.buttons[static_cast<size_t>(state->hoveredButton)].action;
            [self applySettingAction:action];
            [self applyFlowEffects:state->flow.onMenuAction(action)];
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
    if (_savedWorld) return;
    _savedWorld = true;

    EngineState* state = _state.get();
    if (state->saveManager && state->world) {
        state->saveManager->saveMetadata(state->world->getSeed(), state->player.position,
                                         state->worldTime);
        state->saveManager->flush();
        RY_LOG_INFO("World state saved");
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
    if (!state->inputManager || !state->world) return;

    InputState& input = state->inputManager->state();

    // 1. Advance world time (1 tick per game tick)
    state->worldTime++;

    // 2. Hotbar input: keys 1-9 select slots
    for (int i = 0; i < Hotbar::SLOTS; ++i) {
        Key key = static_cast<Key>(static_cast<int>(Key::One) + i);
        if (input.isJustPressed(key)) {
            state->hotbar.selectSlot(i);
            break;
        }
    }

    // 3. Update camera from player state, then consume the look delta so a
    // second tick in the same frame doesn't re-apply it
    state->camera.setPosition(state->player.position);
    InputBindings bindings;
    state->camera.update(state->deltaTime, input, bindings, state->player.position);
    input.clearMouseDelta();

    // 5. Sync player yaw from camera so WASD uses the correct direction
    state->player.yaw = state->camera.yaw();

    // 6. Player physics tick
    bool sprinting = input.isDown(Key::LeftControl);
    state->player.tick(*state->world, input, sprinting);

    // 6b. Footsteps: one thud roughly every two blocks walked on the ground
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

    // 7. Update loaded chunks around player
    state->world->updatePlayerPosition(
        Chunk::worldToChunk(state->player.position.x),
        Chunk::worldToChunk(state->player.position.z));

    // 8. Player jump on space
    if (input.isJustPressed(Key::Space)) {
        state->player.jump();
    }

    // 9. Update weather particles
    if (_renderPipeline) {
        _renderPipeline->tickParticles(state->deltaTime, *state->world, state->player.position);
    }

    // 11. Single raycast for block interaction + highlight
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
    if (input.isJustPressed(Key::MouseLeft)) {
        [self breakBlock:state hit:rayHit];
    }

    // Block placing (right mouse click)
    if (input.isJustPressed(Key::MouseRight)) {
        [self placeBlock:state hit:rayHit];
    }
}

// ---- Block Breaking (Task 6.1) ----

- (void)breakBlock:(EngineState*)state hit:(BlockRayHit)hit {
    if (!hit.has_value()) return;

    int hitX = static_cast<int>(std::floor(hit->first.x));
    int hitY = static_cast<int>(std::floor(hit->first.y));
    int hitZ = static_cast<int>(std::floor(hit->first.z));

    // Cannot break bedrock
    BlockType current = state->world->getBlock(hitX, hitY, hitZ);
    if (current == BlockType::BEDROCK) return;

    // Set block to air
    state->world->setBlock(hitX, hitY, hitZ, BlockType::AIR);

    // Mark chunk dirty
    int chunkX = Chunk::worldToChunk(hitX);
    int chunkZ = Chunk::worldToChunk(hitZ);
    auto chunk = state->world->getChunk(chunkX, chunkZ);
    if (chunk) {
        chunk->markDirty();
    }

    // Trigger save of dirty chunk
    if (state->saveManager && chunk) {
        state->saveManager->saveChunk(*chunk);
    }

    // Clear highlight since block is now air
    state->hasHighlightedBlock = false;

    [self playSfx:state->sfxBlockBreak gain:0.8f];
}

// ---- Block Placing (Task 6.2) ----

- (void)placeBlock:(EngineState*)state hit:(BlockRayHit)hit {
    if (!hit.has_value()) return;

    // Calculate placement position: hit block + face normal
    int placeX = static_cast<int>(std::floor(hit->first.x)) + static_cast<int>(hit->second.x);
    int placeY = static_cast<int>(std::floor(hit->first.y)) + static_cast<int>(hit->second.y);
    int placeZ = static_cast<int>(std::floor(hit->first.z)) + static_cast<int>(hit->second.z);

    // Validate: placement AABB must not overlap player AABB
    AABB placeBox{
        Vec3{static_cast<float>(placeX), static_cast<float>(placeY), static_cast<float>(placeZ)},
        Vec3{static_cast<float>(placeX + 1), static_cast<float>(placeY + 1), static_cast<float>(placeZ + 1)}
    };

    if (placeBox.intersects(state->player.getAABB())) return;

    // Place block
    BlockType selectedType = state->hotbar.getSelectedBlockType();
    state->world->setBlock(placeX, placeY, placeZ, selectedType);

    // Mark both adjacent chunks dirty (boundary case)
    int chunkX = Chunk::worldToChunk(placeX);
    int chunkZ = Chunk::worldToChunk(placeZ);

    auto markChunkDirty = [&](int cx, int cz) {
        auto c = state->world->getChunk(cx, cz);
        if (c) c->markDirty();
    };

    markChunkDirty(chunkX, chunkZ);

    // Check if block is on chunk boundary and mark neighbor
    int localX = placeX - chunkX * CHUNK_WIDTH;
    int localZ = placeZ - chunkZ * CHUNK_DEPTH;
    if (localX == 0) markChunkDirty(chunkX - 1, chunkZ);
    if (localX == CHUNK_WIDTH - 1) markChunkDirty(chunkX + 1, chunkZ);
    if (localZ == 0) markChunkDirty(chunkX, chunkZ - 1);
    if (localZ == CHUNK_DEPTH - 1) markChunkDirty(chunkX, chunkZ + 1);

    // Save affected chunk
    if (state->saveManager) {
        auto chunk = state->world->getChunk(chunkX, chunkZ);
        if (chunk) state->saveManager->saveChunk(*chunk);
    }

    [self playSfx:state->sfxBlockPlace gain:0.8f];
}

// ---- Block Highlight Update (Task 6.9) ----
// Merged into gameTick: via single raycast (Major #4 fix)

// ---- Render ----

- (void)render {
    EngineState* state = _state.get();

    // Update projection matrix from current drawable size
    CGSize currentSize = _view.drawableSize;
    if (currentSize.width > 0 && currentSize.height > 0) {
        state->drawableSize = currentSize;
        float aspect = static_cast<float>(currentSize.width) /
                        static_cast<float>(currentSize.height);
        state->projectionMatrix = Mat4::perspective(
            70.0f * (static_cast<float>(M_PI) / 180.0f),  // 70° FOV in radians
            aspect,
            0.1f,
            1000.0f
        );
    }

    id<CAMetalDrawable> drawable = _view.currentDrawable;
    if (!drawable) return;

    if (!_renderPipeline || !state->world) return;

    // Log render diagnostics every 60 frames
    if (state->frameCount % 60 == 1) {
        auto chunks = state->world->getLoadedChunks();
        RY_LOG_INFO(std::string("Render: ") + std::to_string(chunks.size()) +
            " loaded chunks, frame " + std::to_string(state->frameCount));
    }

    // Playtest hook: RYCRAFT_CAPTURE=<path.png> writes one frame to disk
    // once RYCRAFT_CAPTURE_FRAME (default 240) frames have rendered.
    static const char* capturePath = std::getenv("RYCRAFT_CAPTURE");
    if (capturePath && *capturePath) {
        static const uint64_t captureFrame = [] {
            const char* frameEnv = std::getenv("RYCRAFT_CAPTURE_FRAME");
            return frameEnv ? static_cast<uint64_t>(std::atoll(frameEnv)) : uint64_t{240};
        }();
        if (state->frameCount == captureFrame) {
            _renderPipeline->requestFrameCapture(capturePath);
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
    }

    UIFrameState uiFrame;
    uiFrame.screen = state->flow.screen;
    uiFrame.hoveredButton = state->hoveredButton;
    uiFrame.showDebugHud = state->showDebugHud;
    uiFrame.stats.frameTimeMs = state->smoothedFrameMs;
    uiFrame.stats.fps = state->smoothedFrameMs > 0.f ? 1000.0f / state->smoothedFrameMs : 0.f;
    uiFrame.stats.chunkCount = state->cachedChunkCount;
    uiFrame.stats.entityCount = 0;
    uiFrame.menu = state->menuLayout;

    _renderPipeline->render(
        _queue,
        drawable,
        viewMatrix,
        state->projectionMatrix,
        *state->world,
        state->camera,
        state->worldTime,
        state->hasHighlightedBlock ? std::optional<Vec3>(state->highlightedBlock) : std::nullopt,
        state->hotbar,
        uiFrame
    );
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
    if (!state) return {0, 0};
    return state->drawableSize;
}

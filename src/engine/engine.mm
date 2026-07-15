#import <engine/engine.hpp>

#import <QuartzCore/QuartzCore.h>
#import <mach/mach.h>

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
#include <render/far_terrain.hpp>
#include <render/graphics_settings.hpp>
#include <render/render_pipeline.hpp>
#include <render/ui_menu.hpp>
#include <world/save_manager.hpp>
#include <world/world.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Engine — Singleton (Objective-C class with C++ internals)
// ---------------------------------------------------------------------------

// Raycast hit result type for block interaction
using BlockRayHit = std::optional<std::pair<Vec3, Vec3>>;

struct PerformanceCapture {
    uint64_t warmupFrames = 0;
    uint64_t requestedFrames = 0;
    std::vector<double> frameMilliseconds;
    std::vector<double> fixedTickMilliseconds;
    uint32_t maxLoadedCubes = 0;
    uint32_t maxMeshedCubes = 0;
    size_t maxGenerationQueue = 0;
    uint32_t maxMeshQueue = 0;
    uint32_t maxMeshQueueHighWater = 0;
    uint64_t maxMeshCoalesced = 0;
    uint64_t maxMeshDroppedStale = 0;
    uint32_t maxFarWanted = 0;
    uint32_t maxFarResident = 0;
    uint32_t maxFarDrawn = 0;
    uint32_t maxFarFrustumCulled = 0;
    uint32_t maxFarOcclusionCulled = 0;
    uint32_t maxFarPending = 0;
    uint32_t maxExactUploads = 0;
    uint32_t maxFarUploads = 0;
    float maxFarCacheMB = 0.0f;
    float maxFarArenaMB = 0.0f;
    float maxExactArenaMB = 0.0f;
    float maxGpuFrameMs = 0.0f;
    uint64_t peakResidentBytes = 0;
    uint64_t peakMetalBytes = 0;
    double settleStartSeconds = -1.0;
    double settleSeconds = -1.0;
    bool reported = false;

    bool enabled() const { return requestedFrames > 0; }
};

void recordPerformanceFixedTick(PerformanceCapture& capture, uint64_t frameCount,
                                double milliseconds) {
    if (!capture.enabled() || capture.reported || frameCount < capture.warmupFrames ||
        capture.frameMilliseconds.size() >= capture.requestedFrames) {
        return;
    }
    capture.fixedTickMilliseconds.push_back(milliseconds);
}

uint64_t unsignedEnvironmentValue(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value)
        return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end == value || *end != '\0' ? fallback : static_cast<uint64_t>(parsed);
}

uint64_t processResidentBytes() {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t result = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                           reinterpret_cast<task_info_t>(&info), &count);
    return result == KERN_SUCCESS ? static_cast<uint64_t>(info.resident_size) : 0;
}

bool updatePerformanceCapture(PerformanceCapture& capture, uint64_t frameCount,
                              double frameMilliseconds, uint32_t loadedCubes,
                              uint64_t autopilotStopFrame, World& world, RenderPipeline& renderer,
                              id<MTLDevice> device) {
    if (!capture.enabled() || capture.reported)
        return false;

    const RenderPipeline::ChunkRenderStats stats = renderer.chunkRenderStats();
    if (autopilotStopFrame != std::numeric_limits<uint64_t>::max() &&
        frameCount >= autopilotStopFrame) {
        if (capture.settleStartSeconds < 0.0)
            capture.settleStartSeconds = CACurrentMediaTime();
        const bool settled = world.getPendingChunkCount() == 0 && stats.meshPendingCount == 0 &&
                             stats.farPendingTileCount == 0 &&
                             stats.farResidentTileCount >= stats.farWantedTileCount;
        if (settled && capture.settleSeconds < 0.0) {
            capture.settleSeconds = CACurrentMediaTime() - capture.settleStartSeconds;
        }
    }

    if (frameCount < capture.warmupFrames ||
        capture.frameMilliseconds.size() >= capture.requestedFrames) {
        return false;
    }

    capture.frameMilliseconds.push_back(frameMilliseconds);
    capture.maxLoadedCubes = std::max(capture.maxLoadedCubes, loadedCubes);
    capture.maxMeshedCubes = std::max(capture.maxMeshedCubes, stats.meshCubeCount);
    capture.maxGenerationQueue = std::max(capture.maxGenerationQueue, world.getPendingChunkCount());
    capture.maxMeshQueue = std::max(capture.maxMeshQueue, stats.meshPendingCount);
    capture.maxMeshQueueHighWater =
        std::max(capture.maxMeshQueueHighWater, stats.meshQueueHighWater);
    capture.maxMeshCoalesced = std::max(capture.maxMeshCoalesced, stats.meshCoalescedCount);
    capture.maxMeshDroppedStale =
        std::max(capture.maxMeshDroppedStale, stats.meshDroppedStaleCount);
    capture.maxFarWanted = std::max(capture.maxFarWanted, stats.farWantedTileCount);
    capture.maxFarResident = std::max(capture.maxFarResident, stats.farResidentTileCount);
    capture.maxFarDrawn = std::max(capture.maxFarDrawn, stats.farDrawnTileCount);
    capture.maxFarFrustumCulled =
        std::max(capture.maxFarFrustumCulled, stats.farFrustumCulledTileCount);
    capture.maxFarOcclusionCulled =
        std::max(capture.maxFarOcclusionCulled, stats.farOcclusionCulledTileCount);
    capture.maxFarPending = std::max(capture.maxFarPending, stats.farPendingTileCount);
    capture.maxExactUploads = std::max(capture.maxExactUploads, stats.meshBuildsLastFrame);
    capture.maxFarUploads = std::max(capture.maxFarUploads, stats.farUploadsLastFrame);
    capture.maxFarCacheMB = std::max(capture.maxFarCacheMB, stats.farCacheMB);
    capture.maxFarArenaMB = std::max(capture.maxFarArenaMB, stats.farMegaUsedMB);
    capture.maxExactArenaMB = std::max(capture.maxExactArenaMB, stats.megaUsedMB);
    capture.maxGpuFrameMs = std::max(capture.maxGpuFrameMs, renderer.gpuFrameMs());
    if (frameCount % 30 == 0 || capture.frameMilliseconds.size() == 1) {
        capture.peakResidentBytes = std::max(capture.peakResidentBytes, processResidentBytes());
        capture.peakMetalBytes =
            std::max(capture.peakMetalBytes, static_cast<uint64_t>([device currentAllocatedSize]));
    }

    if (capture.frameMilliseconds.size() < capture.requestedFrames)
        return false;

    std::vector<double> sorted = capture.frameMilliseconds;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double fraction) {
        const size_t index =
            static_cast<size_t>(std::ceil(fraction * static_cast<double>(sorted.size())) - 1.0);
        return sorted[std::min(index, sorted.size() - 1)];
    };
    std::vector<double> prefix(capture.frameMilliseconds.size() + 1, 0.0);
    for (size_t index = 0; index < capture.frameMilliseconds.size(); ++index) {
        prefix[index + 1] = prefix[index] + capture.frameMilliseconds[index];
    }
    double lowestOneSecondFps = std::numeric_limits<double>::infinity();
    for (size_t start = 0; start < capture.frameMilliseconds.size(); ++start) {
        const auto end = std::lower_bound(prefix.begin() + static_cast<ptrdiff_t>(start + 1),
                                          prefix.end(), prefix[start] + 1000.0);
        if (end == prefix.end())
            break;
        const size_t endIndex = static_cast<size_t>(std::distance(prefix.begin(), end));
        const double elapsed = prefix[endIndex] - prefix[start];
        const double fps = static_cast<double>(endIndex - start) * 1000.0 / elapsed;
        lowestOneSecondFps = std::min(lowestOneSecondFps, fps);
    }
    if (!std::isfinite(lowestOneSecondFps))
        lowestOneSecondFps = 1000.0 / percentile(0.95);
    const size_t overTwentyMilliseconds = static_cast<size_t>(
        std::count_if(capture.frameMilliseconds.begin(), capture.frameMilliseconds.end(),
                      [](double milliseconds) { return milliseconds > 20.0; }));
    const double maximumFrame =
        *std::max_element(capture.frameMilliseconds.begin(), capture.frameMilliseconds.end());
    constexpr double MEBIBYTE = 1024.0 * 1024.0;
    const double credibleUnifiedMB =
        static_cast<double>(std::max(capture.peakResidentBytes, capture.peakMetalBytes)) / MEBIBYTE;

    char timing[512];
    std::snprintf(timing, sizeof(timing),
                  "Performance summary: %zu frames p50 %.3f ms p95 %.3f ms max %.3f ms "
                  "lowest 1s %.2f FPS over 20ms %zu gpu EMA max %.3f ms",
                  capture.frameMilliseconds.size(), percentile(0.50), percentile(0.95),
                  maximumFrame, lowestOneSecondFps, overTwentyMilliseconds, capture.maxGpuFrameMs);
    RY_LOG_INFO(timing);

    if (!capture.fixedTickMilliseconds.empty()) {
        std::vector<double> sortedTicks = capture.fixedTickMilliseconds;
        std::sort(sortedTicks.begin(), sortedTicks.end());
        const auto tickPercentile = [&](double fraction) {
            const size_t index = static_cast<size_t>(
                std::ceil(fraction * static_cast<double>(sortedTicks.size())) - 1.0);
            return sortedTicks[std::min(index, sortedTicks.size() - 1)];
        };
        const double maximumTick = *std::max_element(sortedTicks.begin(), sortedTicks.end());
        char fixedTick[256];
        std::snprintf(fixedTick, sizeof(fixedTick),
                      "Performance fixed tick: %zu ticks p50 %.3f ms p95 %.3f ms max %.3f ms",
                      sortedTicks.size(), tickPercentile(0.50), tickPercentile(0.95), maximumTick);
        RY_LOG_INFO(fixedTick);
    } else {
        RY_LOG_INFO("Performance fixed tick: no gameplay ticks in capture window");
    }

    char residency[640];
    std::snprintf(residency, sizeof(residency),
                  "Performance residency: cubes loaded %u meshed %u queues gen %zu mesh %u high %u "
                  "coalesced %llu stale %llu uploads exact %u far wanted %u resident %u drawn %u "
                  "frustum %u occluded %u pending %u uploads %u cache %.1f MB arena %.1f MB exact "
                  "arena %.1f MB",
                  capture.maxLoadedCubes, capture.maxMeshedCubes, capture.maxGenerationQueue,
                  capture.maxMeshQueue, capture.maxMeshQueueHighWater,
                  static_cast<unsigned long long>(capture.maxMeshCoalesced),
                  static_cast<unsigned long long>(capture.maxMeshDroppedStale),
                  capture.maxExactUploads, capture.maxFarWanted, capture.maxFarResident,
                  capture.maxFarDrawn, capture.maxFarFrustumCulled, capture.maxFarOcclusionCulled,
                  capture.maxFarPending, capture.maxFarUploads, capture.maxFarCacheMB,
                  capture.maxFarArenaMB, capture.maxExactArenaMB);
    RY_LOG_INFO(residency);

    const StreamingWorkStats streaming = world.getStreamingWorkStats();
    char planner[384];
    std::snprintf(planner, sizeof(planner),
                  "Performance streaming planner: rebuilds %llu requests %llu coalesced %llu "
                  "canceled %llu build EMA %.3f ms notifications %llu",
                  static_cast<unsigned long long>(streaming.activeSetRebuilds),
                  static_cast<unsigned long long>(streaming.activeSetRequests),
                  static_cast<unsigned long long>(streaming.activeSetRequestsCoalesced),
                  static_cast<unsigned long long>(streaming.activeSetBuildsCanceled),
                  streaming.activeSetBuildMs,
                  static_cast<unsigned long long>(streaming.activeSetRebuildNotifications));
    RY_LOG_INFO(planner);

    char memory[384];
    std::snprintf(memory, sizeof(memory),
                  "Performance memory: process RSS %.1f MB Metal allocated %.1f MB credible "
                  "unified %.1f MB queue settle %.3f s",
                  static_cast<double>(capture.peakResidentBytes) / MEBIBYTE,
                  static_cast<double>(capture.peakMetalBytes) / MEBIBYTE, credibleUnifiedMB,
                  capture.settleSeconds);
    RY_LOG_INFO(memory);
    capture.reported = true;
    return true;
}

// Internal C++ state
struct EngineState {
    // ---- Game loop state ----
    double lastTime = 0;
    double deltaTime = 0;
    uint64_t frameCount = 0;
    uint64_t autoPauseFrame = std::numeric_limits<uint64_t>::max();

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
    PerformanceCapture performance;
    uint64_t autopilotStartFrame = 0;
    uint64_t autopilotStopFrame = std::numeric_limits<uint64_t>::max();

    // ---- Player & World ----
    Player player;
    std::shared_ptr<World> world;
    std::unique_ptr<SaveManager> saveManager;
    Camera camera;

    // ---- Animals ----
    std::unique_ptr<Spawner> spawner;
    std::unordered_map<uint64_t, StateMachine> entityBrains;
    int animalCallCooldown = 0;

    // ---- Audio ----
    std::unique_ptr<AudioEngine> audio;
    std::vector<float> sfxBlockBreak;
    std::vector<float> sfxBlockPlace;
    std::vector<float> sfxFootstep;
    std::vector<float> sfxWind;
    std::array<std::vector<float>, ENTITY_TYPE_COUNT> sfxAnimal;
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
        _state->performance.requestedFrames =
            std::min<uint64_t>(unsignedEnvironmentValue("RYCRAFT_PERF_FRAMES", 0), 36'000);
        _state->performance.warmupFrames =
            unsignedEnvironmentValue("RYCRAFT_PERF_WARMUP_FRAMES", 600);
        if (_state->performance.enabled()) {
            _state->performance.frameMilliseconds.reserve(
                static_cast<size_t>(_state->performance.requestedFrames));
            _state->performance.fixedTickMilliseconds.reserve(
                static_cast<size_t>(_state->performance.requestedFrames));
        }
        _state->autopilotStartFrame = unsignedEnvironmentValue("RYCRAFT_AUTOPILOT_START_FRAME", 0);
        _state->autopilotStopFrame = unsignedEnvironmentValue("RYCRAFT_AUTOPILOT_STOP_FRAME",
                                                              std::numeric_limits<uint64_t>::max());
        _state->autoPauseFrame = unsignedEnvironmentValue("RYCRAFT_AUTOPAUSE_FRAME",
                                                          std::numeric_limits<uint64_t>::max());
        _state->saveManager = std::make_unique<SaveManager>([@"rycraft_world" UTF8String]);

        // Resume the saved world when one exists; otherwise start fresh
        uint32_t seed = 42;
        Vec3 spawnPos{0.f, 100.f, 0.f};
        if (auto meta = _state->saveManager->loadMetadata()) {
            seed = meta->seed;
            spawnPos = meta->spawnPos;
            _state->worldTime = meta->worldTime;
            _state->player.yaw = meta->player.yaw;
            _state->player.pitch = meta->player.pitch;
            _state->player.health = meta->player.health;
            _state->hotbar.selectSlot(meta->player.selectedSlot);
            for (int slot = 0; slot < Hotbar::SLOTS; ++slot) {
                _state->hotbar.setSlot(slot, meta->player.inventory[static_cast<size_t>(slot)]);
            }
        }
        if (const char* seedEnv = std::getenv("RYCRAFT_WORLD_SEED")) {
            seed = static_cast<uint32_t>(std::strtoull(seedEnv, nullptr, 0));
        }
        if (const char* spawnEnv = std::getenv("RYCRAFT_SPAWN")) {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            if (std::sscanf(spawnEnv, "%f,%f,%f", &x, &y, &z) == 3)
                spawnPos = {x, y, z};
        }

        // Playtest hook: pin the time of day (0..23999; 6000 = noon).
        if (const char* timeEnv = std::getenv("RYCRAFT_TIME")) {
            _state->worldTime = static_cast<uint64_t>(std::clamp(std::atoi(timeEnv), 0, 23999));
        }

        // Persisted settings load before the World exists (view distance
        // feeds its constructor); env overrides win over the file for
        // headless playtests. Playtest hook: RYCRAFT_VIEW_DISTANCE=<4..256>.
        // An env-overridden session never saves settings — a playtest run
        // must not rewrite the user's file with its overrides.
        LoadedSettings loaded = loadSettings(settingsPath());
        _state->settings = loaded.values;
        _state->gfx = loaded.gfx;
        _state->envOverridesActive = _state->gfx.applyEnvOverrides();
        if (const char* vdEnv = std::getenv("RYCRAFT_VIEW_DISTANCE")) {
            _state->settings.viewDistance =
                std::clamp(std::atoi(vdEnv), SettingsValues::MIN_VIEW_DISTANCE,
                           SettingsValues::MAX_VIEW_DISTANCE);
            _state->envOverridesActive = true;
        }
        if (const char* overlayEnv = std::getenv("RYCRAFT_WORLDGEN_OVERLAY")) {
            _state->showDebugHud = *overlayEnv != '\0';
        }
        if (const char* debugEnv = std::getenv("RYCRAFT_SHOW_DEBUG")) {
            _state->showDebugHud = *debugEnv != '\0' && std::strcmp(debugEnv, "0") != 0;
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
    const bool nativeWindow = [] {
        const char* value = std::getenv("RYCRAFT_NATIVE_WINDOW");
        return value && *value && std::strcmp(value, "0") != 0;
    }();
    NSRect screenFrame =
        nativeWindow ? [[NSScreen mainScreen] frame] : [[NSScreen mainScreen] visibleFrame];
    const CGFloat windowWidth = nativeWindow ? screenFrame.size.width : 1024.0;
    const CGFloat windowHeight = nativeWindow ? screenFrame.size.height : 768.0;
    NSRect windowRect =
        NSMakeRect(screenFrame.origin.x + (screenFrame.size.width - windowWidth) * 0.5,
                   screenFrame.origin.y + (screenFrame.size.height - windowHeight) * 0.5,
                   windowWidth, windowHeight);
    const NSWindowStyleMask styleMask =
        nativeWindow ? NSWindowStyleMaskBorderless
                     : NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                           NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
    _window = [[NSWindow alloc] initWithContentRect:windowRect
                                          styleMask:styleMask
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
    _view = [[MTKView alloc] initWithFrame:NSMakeRect(0.0, 0.0, windowWidth, windowHeight)
                                    device:_device];
    if (!_view) {
        RY_LOG_FATAL("Failed to create MTKView");
        return NO;
    }

    // Drawable pixel format. The render pipeline builds its own MSAA render
    // passes, so the view carries no sample count or depth buffer of its own.
    _view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    _view.preferredFramesPerSecond = 120;

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
        for (size_t index = 0; index < ENTITY_TYPE_COUNT; ++index) {
            _state->sfxAnimal[index] =
                SoundEffect::generateAnimalCall(static_cast<EntityType>(index));
        }
        [self syncAudioVolume];
    } else {
        RY_LOG_ERROR("Audio engine failed to initialize — continuing without sound");
        _state->audio.reset();
    }

    RY_LOG_INFO(std::string("Engine initialized - window: ") +
                std::to_string(static_cast<int>(_view.bounds.size.width)) + "x" +
                std::to_string(static_cast<int>(_view.bounds.size.height)) +
                ", drawable: " + std::to_string(static_cast<int>(_view.drawableSize.width)) + "x" +
                std::to_string(static_cast<int>(_view.drawableSize.height)) +
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

    // Performance hook: freeze the exact same streamed scene at a fixed frame
    // so playing and paused windows can be compared without relying on a
    // synthetic key event or a separately generated world.
    if (state->frameCount == state->autoPauseFrame && state->flow.screen == GameScreen::PLAYING) {
        state->flow.screen = GameScreen::PAUSED;
        state->accumulator = 0.0;
    }

    // 4. Fixed timestep game tick — menus freeze the world
    if (state->flow.screen == GameScreen::PLAYING) {
        while (state->accumulator >= EngineState::TICK_DT) {
            const double tickStart = CACurrentMediaTime();
            [self gameTick:state];
            recordPerformanceFixedTick(state->performance, state->frameCount,
                                       (CACurrentMediaTime() - tickStart) * 1000.0);
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
            constexpr const auto& distances = SettingsValues::VIEW_DISTANCES;
            const bool increase = action == MenuAction::VIEW_DISTANCE_UP;
            auto current =
                std::lower_bound(distances.begin(), distances.end(), settings.viewDistance);
            if (increase) {
                if (current == distances.end()) {
                    current = std::prev(distances.end());
                } else if (*current <= settings.viewDistance) {
                    const auto next = std::next(current);
                    if (next != distances.end())
                        current = next;
                }
            } else {
                if (current == distances.end() || *current >= settings.viewDistance) {
                    if (current != distances.begin())
                        --current;
                }
            }
            settings.viewDistance = *current;
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
    _savedWorld = true;

    // Render workers retain references into World. Join them for every quit
    // mode, including throwaway capture runs that intentionally skip saves.
    if (_renderPipeline) {
        _renderPipeline->shutdownMeshWorkers();
    }

    // Capture runs are throwaway playtests: their spawned test blocks
    // (RYCRAFT_SPAWN_*) and drifted player position must not overwrite the
    // real save, and reproducible captures depend on the world not moving.
    if (std::getenv("RYCRAFT_CAPTURE")) {
        return;
    }

    EngineState* state = _state.get();
    if (state->saveManager && state->world) {
        // Edited chunks persist on unload; the quit path sweeps the rest
        const bool frontiersSaved = state->world->saveModifiedChunks();
        SaveManager::PlayerMetadata playerMetadata;
        playerMetadata.yaw = state->player.yaw;
        playerMetadata.pitch = state->player.pitch;
        playerMetadata.health = state->player.health;
        playerMetadata.selectedSlot = state->hotbar.getSelectedIndex();
        for (int slot = 0; slot < Hotbar::SLOTS; ++slot) {
            playerMetadata.inventory[static_cast<size_t>(slot)] = state->hotbar.getSlot(slot);
        }
        const bool metadataSaved = state->saveManager->saveMetadata(
            state->world->getSeed(), state->player.position, state->worldTime, playerMetadata);
        const bool cubesSaved = state->saveManager->flush();
        if (frontiersSaved && metadataSaved && cubesSaved) {
            RY_LOG_INFO("World state saved");
        } else {
            RY_LOG_ERROR("World state save did not complete");
        }
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
    // Automated movement must keep exercising streaming when the launcher's
    // terminal regains focus. Release the global pointer lock while leaving
    // the simulation on the playing screen; gameTick restores only the
    // synthetic movement keys on its next fixed tick.
    if (std::getenv("RYCRAFT_AUTOPILOT")) {
        if (_state->inputManager) {
            _state->inputManager->releaseMouse();
        }
        return;
    }
    // Cmd-Tab (or any focus loss) must never leave the pointer locked
    [self applyFlowEffects:_state->flow.onFocusLost()];
}

// ---- Game Tick (fixed timestep at 20Hz) ----

- (void)gameTick:(EngineState*)state {
    if (!state->inputManager || !state->world)
        return;

    InputState& input = state->inputManager->state();

    // Playtest hook: walk and sprint exercise ground physics. Fly holds a
    // level aerial route above obstacles so streaming benchmarks cross many
    // chunk boundaries without depending on one terrain fixture.
    static const char* autopilot = std::getenv("RYCRAFT_AUTOPILOT");
    if (autopilot) {
        const std::string_view mode{autopilot};
        const bool active = state->frameCount >= state->autopilotStartFrame &&
                            state->frameCount < state->autopilotStopFrame;
        const bool aerial = mode == "fly";
        if (aerial)
            state->player.flying = true;
        input.keysDown[Key::W] = active && (mode == "walk" || mode == "sprint" || aerial);
        input.keysDown[Key::LeftControl] = active && (mode == "sprint" || aerial);
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
        int64_t px = static_cast<int64_t>(std::floor(state->player.position.x));
        int64_t pz = static_cast<int64_t>(std::floor(state->player.position.z));
        int32_t feetY = static_cast<int32_t>(std::floor(state->player.position.y));
        auto chunk = state->world->getChunk(Chunk::worldToChunk(px), Chunk::worldToChunkY(feetY),
                                            Chunk::worldToChunk(pz));
        if (chunk && chunk->generated) {
            bool embedded = isSolid(state->world->getBlock(px, feetY, pz)) ||
                            isSolid(state->world->getBlock(px, feetY + 1, pz));
            if (embedded) {
                int32_t surfaceY = static_cast<int32_t>(state->world->getTerrainHeight(px, pz));
                state->player.position.y = static_cast<float>(surfaceY + 2);
                state->player.velocity = Vec3{0.f, 0.f, 0.f};
                RY_LOG_INFO("Spawn was inside terrain and moved to the surface");
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
                        int64_t wx = px + dx;
                        int64_t wz = pz + 6 + dz; // ahead (+Z) at spawn
                        int32_t searchTop = static_cast<int32_t>(
                            std::clamp(state->world->getTerrainHeight(wx, wz) + CHUNK_EDGE,
                                       static_cast<double>(WORLD_MIN_Y + 1),
                                       static_cast<double>(WORLD_MAX_Y - 1)));
                        for (int32_t y = searchTop; y > WORLD_MIN_Y; --y) {
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
                           b == BlockType::SPRUCE_LOG || b == BlockType::ACACIA_LOG ||
                           b == BlockType::JUNGLE_LOG || b == BlockType::MANGROVE_LOG ||
                           b == BlockType::PALM_LOG || b == BlockType::WILLOW_LOG;
                };
                for (int dz = 3; dz <= 22; ++dz) {
                    for (int dx = -10; dx <= 10; ++dx) {
                        int64_t wx = px + dx;
                        int64_t wz = pz + dz;
                        // Scan down to the terrain surface, past any tree trunk,
                        // so the pool sits at ground level (no floating water).
                        int32_t searchTop = static_cast<int32_t>(
                            std::clamp(state->world->getTerrainHeight(wx, wz) + CHUNK_EDGE,
                                       static_cast<double>(WORLD_MIN_Y + 1),
                                       static_cast<double>(WORLD_MAX_Y - 1)));
                        for (int32_t y = searchTop; y > WORLD_MIN_Y; --y) {
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
    const Vec3 playerPositionBeforeMove = state->player.position;
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
    state->world->updatePlayerPosition(static_cast<int64_t>(std::floor(state->player.position.x)),
                                       static_cast<int32_t>(std::floor(state->player.position.y)),
                                       static_cast<int64_t>(std::floor(state->player.position.z)));
    state->world->tickFluids(EngineState::TICK_DT);

    // 6b. Animals: initial population once the spawn area streamed in, then
    // per-tick AI steering + physics for every living entity
    if (state->spawner) {
        state->spawner->updatePopulation(state->player.position);

        auto& entities = state->spawner->getEntities();
        auto& spatialHash = state->spawner->getSpatialHash();
        std::unordered_set<uint64_t> liveEntityIds;
        liveEntityIds.reserve(entities.size());
        for (const auto& entity : entities) {
            if (entity)
                liveEntityIds.insert(entity->id);
        }
        std::erase_if(state->entityBrains,
                      [&](const auto& entry) { return !liveEntityIds.contains(entry.first); });
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
            const bool playerMovingToward = StateMachine::playerMovedToward(
                playerPositionBeforeMove, state->player.position, entity->position);
            Vec3 steering =
                brain.update(*entity, *state->world, state->player.position, playerMovingToward,
                             /*playerHoldingFood=*/false, *state->spawner);
            entity->velocity.x += steering.x;
            entity->velocity.y += steering.y;
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

    // Publish the immutable loaded-cube registry once per simulation tick.
    // Rendering reads this pointer without copying or taking the world lock.
    state->world->publishLoadedSnapshot();

    // Tick-edge input consumed — a second tick in this frame must not re-fire
    input.clearTickPresses();
}

// ---- Block Breaking (Task 6.1) ----

- (void)breakBlock:(EngineState*)state hit:(BlockRayHit)hit {
    if (!hit.has_value())
        return;

    int64_t hitX = static_cast<int64_t>(std::floor(hit->first.x));
    int32_t hitY = static_cast<int32_t>(std::floor(hit->first.y));
    int64_t hitZ = static_cast<int64_t>(std::floor(hit->first.z));

    // Cannot break bedrock
    const std::optional<BlockType> current = state->world->findBlockIfLoaded(hitX, hitY, hitZ);
    if (!current || *current == BlockType::BEDROCK)
        return;

    // Set block to air
    state->world->setBlock(hitX, hitY, hitZ, BlockType::AIR);

    // Flora standing on the broken block loses its support and pops with it
    // (same column → same chunk → the dirty/save below covers it)
    const std::optional<BlockType> flora = state->world->findBlockIfLoaded(hitX, hitY + 1, hitZ);
    if (flora && isFlora(*flora)) {
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
    int64_t placeX =
        static_cast<int64_t>(std::floor(hit->first.x)) + static_cast<int64_t>(hit->second.x);
    int32_t placeY =
        static_cast<int32_t>(std::floor(hit->first.y)) + static_cast<int32_t>(hit->second.y);
    int64_t placeZ =
        static_cast<int64_t>(std::floor(hit->first.z)) + static_cast<int64_t>(hit->second.z);

    // Validate: placement AABB must not overlap player AABB
    AABB placeBox{
        Vec3{static_cast<float>(placeX), static_cast<float>(placeY), static_cast<float>(placeZ)},
        Vec3{static_cast<float>(placeX + 1), static_cast<float>(placeY + 1),
             static_cast<float>(placeZ + 1)}};

    if (placeBox.intersects(state->player.getAABB()))
        return;

    const ChunkPos placeChunk{Chunk::worldToChunk(placeX), Chunk::worldToChunkY(placeY),
                              Chunk::worldToChunk(placeZ)};
    if (!state->world->isChunkLoaded(placeChunk))
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
        const float farPlane =
            std::max(1000.0f, static_cast<float>(state->settings.viewDistance * CHUNK_EDGE +
                                                 FAR_TERRAIN_TILE_EDGE * 2));
        state->projectionMatrix = Mat4::perspective(
            state->camera.FOV() * (static_cast<float>(M_PI) / 180.0f), aspect, 0.1f, farPlane);
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
        const StreamingWorkStats streaming = state->world->getStreamingWorkStats();
        char line[768];
        snprintf(line, sizeof(line),
                 "Render: frame %llu player (%.1f, %.1f, %.1f) | %.2f ms/frame gpu %.2f ms "
                 "cubes %u loaded %u meshed gen %.2f ms mesh %.2f ms queues %zu/%u high %u "
                 "exact %.0f/%.0f MB far %u wanted %u resident %u drawn %u frustum %u occluded "
                 "%u pending %.0f MB cache %.0f MB arena planner %.1f ms %llu/%llu/%llu",
                 static_cast<unsigned long long>(state->frameCount), state->player.position.x,
                 state->player.position.y, state->player.position.z, state->smoothedFrameMs,
                 _renderPipeline->gpuFrameMs(), state->cachedChunkCount, chunkStats.meshCubeCount,
                 state->world->averageGenMs(), chunkStats.meshMsAvg,
                 state->world->getPendingChunkCount(), chunkStats.meshPendingCount,
                 chunkStats.meshQueueHighWater, chunkStats.megaUsedMB, chunkStats.megaCapMB,
                 chunkStats.farWantedTileCount, chunkStats.farResidentTileCount,
                 chunkStats.farDrawnTileCount, chunkStats.farFrustumCulledTileCount,
                 chunkStats.farOcclusionCulledTileCount, chunkStats.farPendingTileCount,
                 chunkStats.farCacheMB, chunkStats.farMegaUsedMB, streaming.activeSetBuildMs,
                 static_cast<unsigned long long>(streaming.activeSetRequests),
                 static_cast<unsigned long long>(streaming.activeSetRequestsCoalesced),
                 static_cast<unsigned long long>(streaming.activeSetBuildsCanceled));
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
        state->cachedChunkCount = static_cast<uint32_t>(state->world->getLoadedChunkCount());
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
        const int64_t waterX = static_cast<int64_t>(std::floor(camPos.x));
        const int32_t waterY = static_cast<int32_t>(std::floor(camPos.y));
        const int64_t waterZ = static_cast<int64_t>(std::floor(camPos.z));
        uiFrame.cameraUnderwater =
            state->world->getBlockIfLoaded(waterX, waterY, waterZ) == BlockType::WATER &&
            camPos.y < static_cast<float>(waterY) +
                           state->world->getFluidHeightIfLoaded(waterX, waterY, waterZ);
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
    uiFrame.stats.meshedCubeCount = chunkStats.meshCubeCount;
    uiFrame.stats.farWantedTiles = chunkStats.farWantedTileCount;
    uiFrame.stats.farResidentTiles = chunkStats.farResidentTileCount;
    uiFrame.stats.farDrawnTiles = chunkStats.farDrawnTileCount;
    uiFrame.stats.farFrustumCulledTiles = chunkStats.farFrustumCulledTileCount;
    uiFrame.stats.farOcclusionCulledTiles = chunkStats.farOcclusionCulledTileCount;
    uiFrame.stats.farPendingTiles = chunkStats.farPendingTileCount;
    uiFrame.stats.farCacheMB = chunkStats.farCacheMB;
    uiFrame.stats.farMeshMB = chunkStats.farMegaUsedMB;
    const int64_t playerX = static_cast<int64_t>(std::floor(state->player.position.x));
    const int32_t playerY = static_cast<int32_t>(std::floor(state->player.position.y));
    const int64_t playerZ = static_cast<int64_t>(std::floor(state->player.position.z));
    uiFrame.stats.cubeX = Chunk::worldToChunk(playerX);
    uiFrame.stats.cubeY = Chunk::worldToChunkY(playerY);
    uiFrame.stats.cubeZ = Chunk::worldToChunk(playerZ);
    const size_t planEntries = state->world->cachedColumnPlanCount();
    const worldgen::BasinCacheMetrics basinCache = state->world->generator().basinCacheMetrics();
    uiFrame.stats.macroCacheEntries = static_cast<uint32_t>(planEntries + basinCache.entries);
    uiFrame.stats.macroCacheMB =
        static_cast<float>(planEntries * sizeof(ColumnPlan) + basinCache.bytes) /
        (1024.0f * 1024.0f);
    uiFrame.stats.pendingFluids = static_cast<uint32_t>(state->world->getPendingFluidCount());
    uiFrame.stats.droppedFluidUpdates = state->world->getDroppedFluidUpdateCount();
    uiFrame.stats.droppedFluidFrontiers = state->world->getDroppedFluidFrontierCount();
    if (const auto surface = state->world->findSurfaceSample(playerX, playerZ)) {
        uiFrame.stats.plateId = surface->geology.plateId;
        uiFrame.stats.boundary = surface->geology.boundary;
        uiFrame.stats.temperatureC = static_cast<float>(surface->climate.temperatureC);
        uiFrame.stats.precipitationMm = static_cast<float>(surface->climate.annualPrecipitationMm);
        uiFrame.stats.primaryBiome = surface->biome.primary;
        uiFrame.stats.secondaryBiome = surface->biome.secondary;
        uiFrame.stats.biomeTransition = static_cast<float>(surface->biome.transition);
        uiFrame.stats.riverOrder = surface->hydrology.streamOrder;
    }
    uiFrame.menu = state->menuLayout;

    _renderPipeline->render(
        _queue, drawable, viewMatrix, state->projectionMatrix, *state->world, state->camera,
        state->worldTime,
        state->hasHighlightedBlock ? std::optional<Vec3>(state->highlightedBlock) : std::nullopt,
        state->hotbar, uiFrame, state->spawner ? &state->spawner->getEntities() : nullptr);

    if (updatePerformanceCapture(state->performance, state->frameCount, state->deltaTime * 1000.0,
                                 state->cachedChunkCount, state->autopilotStopFrame, *state->world,
                                 *_renderPipeline, _device)) {
        [self requestQuit];
    }
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

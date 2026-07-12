#import <engine/engine.hpp>

#import <QuartzCore/QuartzCore.h>

#include <common/error.hpp>
#include <common/math.hpp>
#include <render/render_pipeline.hpp>
#include <engine/camera.hpp>
#include <engine/hotbar.hpp>
#include <engine/input_bindings.hpp>
#include <entity/player.hpp>
#include <entity/voxel_traversal.hpp>
#include <world/world.hpp>
#include <world/save_manager.hpp>

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

    // ---- Player & World ----
    Player player;
    std::shared_ptr<World> world;
    std::unique_ptr<SaveManager> saveManager;
    Camera camera;

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
        _state->world = std::make_shared<World>(42);
        _state->saveManager = std::make_unique<SaveManager>([@"rycraft_world" UTF8String]);
        // Spawn player above terrain
        _state->player.position = Vec3{0.f, 100.f, 0.f};
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

    // Pixel formats
    _view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    _view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    _view.sampleCount = 4;  // 4x MSAA

    // Note: MetalFX upscaling will be configured in Phase 8.6.
    // The depth stencil format is set to Depth32Float to support
    // MetalFX's depth-based temporal upscaling requirements.

    // Disable automatic setNeedsDisplay — we drive rendering from the game loop
    _view.enableSetNeedsDisplay = false;
    _view.framebufferOnly = false;

    // Delegate
    _view.delegate = self;

    // Set as window content
    [_window setContentView: _view];

    // 7. Create InputManager
    _state->inputManager = new InputManager(_window);
    _state->inputManager->hideAndConfineCursor();

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

    // 3. Fixed timestep game tick
    while (state->accumulator >= EngineState::TICK_DT) {
        [self gameTick:state];
        state->accumulator -= EngineState::TICK_DT;
    }

    // 4. Render
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

    // 3. Hotbar input: scroll wheel cycles slots
    _scrollAccumulator += input.mouseDelta.y;
    while (_scrollAccumulator >= 30.0f) {
        state->hotbar.selectNext();
        _scrollAccumulator -= 30.0f;
    }
    while (_scrollAccumulator <= -30.0f) {
        state->hotbar.selectPrev();
        _scrollAccumulator += 30.0f;
    }

    // 4. Update camera from player state
    state->camera.setPosition(state->player.position);
    InputBindings bindings;
    state->camera.update(state->deltaTime, input, bindings, state->player.position);

    // 5. Player physics tick
    bool sprinting = input.isDown(Key::LeftControl);
    state->player.tick(*state->world, input, sprinting);

    // 6. Player jump on space
    if (input.isJustPressed(Key::Space)) {
        state->player.jump();
    }

    // 7. Single raycast for block interaction + highlight
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

    // Camera view matrix
    Mat4 viewMatrix = state->camera.viewMatrix();

    _renderPipeline->render(
        _queue,
        drawable,
        viewMatrix,
        state->projectionMatrix,
        *state->world,
        state->camera,
        state->worldTime,
        state->hasHighlightedBlock ? std::optional<Vec3>(state->highlightedBlock) : std::nullopt,
        state->hotbar
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

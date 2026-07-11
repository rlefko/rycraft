#import <engine/engine.hpp>

#import <QuartzCore/QuartzCore.h>

#include <common/error.hpp>
#include <common/math.hpp>

#include <chrono>
#include <memory>

// ---------------------------------------------------------------------------
// Engine — Singleton (Objective-C class with C++ internals)
// ---------------------------------------------------------------------------

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
};

@implementation Engine {
    NSApplication* _app;
    NSWindow* _window;
    MTKView* _view;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;

    // C++ game state
    std::unique_ptr<EngineState> _state;
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
        _state = std::make_unique<EngineState>();
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

    // Disable automatic setNeedsDisplay — we drive rendering from the game loop
    _view.enableSetNeedsDisplay = false;
    _view.framebufferOnly = false;

    // Delegate
    _view.delegate = self;

    // Set as window content
    [_window setContentView: _view];

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

    // 3. Fixed timestep physics
    while (state->accumulator >= EngineState::TICK_DT) {
        // physicsTick() — will be filled in Phase 5
        state->accumulator -= EngineState::TICK_DT;
    }

    // 4. Render
    [self render];

    // 5. Increment frame count
    state->frameCount++;

    (void)view;
}

- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)newSize {
    // Projection matrix is recomputed in render() each frame.
    (void)view;
    (void)newSize;
}

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

    // Render pipeline — will be filled in Phase 4
    // For now, create and present an empty command buffer to keep the view alive
    id<CAMetalDrawable> drawable = _view.currentDrawable;
    if (!drawable) return;

    id<MTLCommandBuffer> commandBuffer = [_queue commandBuffer];
    if (!commandBuffer) return;

    [commandBuffer presentDrawable: drawable];
    [commandBuffer commit];
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

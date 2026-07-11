engine_sources = files(
    'engine.mm',
    'input.cpp',
    'input_bindings.cpp',
    'camera.cpp',
)

libengine = static_library('engine',
    engine_sources,
    include_directories : inc,
    dependencies : [appleframeworks],
)

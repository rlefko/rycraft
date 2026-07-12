engine_sources = files(
    'engine.mm',
    'input.mm',
    'input_bindings.cpp',
    'camera.cpp',
    'hotbar.cpp',
)

libengine = static_library('engine',
    engine_sources,
    include_directories : inc,
    dependencies : [appleframeworks],
)

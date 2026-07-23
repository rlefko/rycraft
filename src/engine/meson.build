engine_sources = files(
    'engine.mm',
    'input.mm',
    'input_bindings.cpp',
    'camera.cpp',
    'inventory.cpp',
    'slot_interaction.cpp',
    'mining.cpp',
    'survival.cpp',
    'v4_world_startup.cpp',
)

libengine = static_library('engine',
    engine_sources,
    include_directories : inc,
    dependencies : [appleframeworks],
)

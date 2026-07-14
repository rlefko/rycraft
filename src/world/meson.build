world_sources = files(
  'world.cpp',
  'noise.cpp',
  'climate.cpp',
  'density_field.cpp',
  'chunk_generator.cpp',
  'ores.cpp',
  'structures.cpp',
  'features.cpp',
  'chunk.cpp',
  'light_engine.cpp',
  'serialization.cpp',
  'save_manager.cpp',
)

libworld = static_library('world',
  world_sources,
  include_directories : inc,
  dependencies : [appleframeworks, lz4_dep],
)

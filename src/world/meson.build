world_sources = files(
  'world.cpp',
  'noise.cpp',
  'terrain.cpp',
  'biome.cpp',
  'chunk.cpp',
  'surface.cpp',
  'caves.cpp',
  'ores.cpp',
  'trees.cpp',
  'structures.cpp',
  'serialization.cpp',
  'save_manager.cpp',
)

libworld = static_library('world',
  world_sources,
  include_directories : inc,
  dependencies : [appleframeworks, lz4_dep],
)

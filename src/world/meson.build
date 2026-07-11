lz4_dep = dependency('lz4', required : false)
if not lz4_dep.found()
  lz4_proj = subproject('lz4')
  lz4_dep = lz4_proj.get_variable('liblz4_dep')
endif

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

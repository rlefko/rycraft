entity_sources = files(
  'entity.cpp',
  'physics.cpp',
  'player.cpp',
  'voxel_traversal.cpp',
)

libentity = static_library('entity',
  entity_sources,
  include_directories : inc,
  dependencies : [appleframeworks],
)

entity_sources = files(
  'entity.cpp',
  'physics.cpp',
  'player.cpp',
  'voxel_traversal.cpp',
  'ai.cpp',
  'spatial_hash.cpp',
  'spawner.cpp',
  'item_entity.cpp',
)

libentity = static_library('entity',
  entity_sources,
  include_directories : inc,
  dependencies : [appleframeworks],
)

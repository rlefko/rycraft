render_sources = files('render.cpp', 'lod_mesher.cpp', 'block_texture_array.mm', 'mega_buffer.mm', 'render_pipeline.mm', 'ui_overlay.mm', 'ui_menu.cpp', 'ui_hud.mm', 'entity_renderer.mm', 'bloom.mm', 'particles.mm')

librender = static_library('render',
  render_sources,
  include_directories : inc,
  dependencies : [appleframeworks],
)

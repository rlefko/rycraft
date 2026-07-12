render_sources = files('render.cpp', 'mesher.cpp', 'texture_atlas.mm', 'mega_buffer.mm', 'render_pipeline.mm', 'ui_overlay.mm', 'bloom.mm')

librender = static_library('render',
  render_sources,
  include_directories : inc,
  dependencies : [appleframeworks],
)

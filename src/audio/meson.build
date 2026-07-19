audio_sources = files('audio_engine.mm', 'sfx.cpp', 'thunder.cpp')

libaudio = static_library('audio',
  audio_sources,
  include_directories : inc,
  dependencies : [appleframeworks],
)

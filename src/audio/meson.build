audio_sources = files('audio.cpp', 'audio_engine.mm', 'sfx.cpp')

libaudio = static_library('audio',
  audio_sources,
  include_directories : inc,
  dependencies : [appleframeworks],
)

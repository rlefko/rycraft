subdir('common')
subdir('engine')
subdir('render')
subdir('world')
subdir('entity')
subdir('audio')

# Copy compiled metallib next to the executable so it's discoverable at runtime
copy_metallib = configure_file(
  input : '../shaders/pipeline.metallib',
  output : 'pipeline.metallib',
  configuration : {},
)

rycraft_exe = executable('rycraft',
  'engine/main.cpp',
  link_with : [
    libcommon,
    libengine,
    librender,
    libworld,
    libentity,
    libaudio,
  ],
  dependencies : [appleframeworks],
  depends : [metallib],
)

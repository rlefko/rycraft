subdir('common')
subdir('engine')
subdir('render')
subdir('world')
subdir('entity')
subdir('audio')

# Copy compiled metallib next to the executable so it's discoverable at runtime
copy_metallib = custom_target(
  'copy_metallib',
  input : metallib,
  output : 'pipeline.metallib',
  command : ['cp', '@INPUT@', '@OUTPUT@'],
  build_by_default : true,
)

rycraft_exe = executable('rycraft',
  'engine/main.mm',
  include_directories : inc,
  link_with : [
    libcommon,
    libengine,
    librender,
    libworld,
    libentity,
    libaudio,
  ],
  dependencies : [appleframeworks],
  build_by_default : true,
)

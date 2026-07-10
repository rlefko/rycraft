subdir('common')
subdir('engine')
subdir('render')
subdir('world')
subdir('entity')
subdir('audio')

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
)

# Metal shader compilation pipeline: each .metal -> .air, then all .air -> one .metallib

shader_names = ['pipeline', 'ui_overlay', 'sky', 'bloom', 'clouds', 'particles', 'post', 'exposure',
                'shadow', 'volumetrics', 'ssao']

air_targets = []
foreach name : shader_names
  air_targets += custom_target(
    name + '.air',
    input : name + '.metal',
    output : name + '.air',
    command : [
      'xcrun', 'metal',
      '-c',
      '-mmacos-version-min=13.0',
      '-std=metal3.0',
      '-I', meson.project_source_root() / 'include',
      '-o', '@OUTPUT@',
      '@INPUT@',
    ],
    depend_files : files('../include/render/shader_types.hpp'),
  )
endforeach

metallib = custom_target(
  'pipeline.metallib',
  input : air_targets,
  output : 'pipeline.metallib',
  command : ['xcrun', 'metallib', '@INPUT@', '-o', '@OUTPUT@'],
  build_by_default : true,
)

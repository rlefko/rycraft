project('rycraft', 'cpp', 'objcpp',
  version : '0.1.0',
  default_options : [
    'cpp_std=c++23',
    'warning_level=3',
    'werror=true',
  ],
)

cpp = meson.get_compiler('cpp')

# Central include directory (shared across all subdirs)
inc = include_directories('include')

# Apple frameworks
appleframeworks = dependency('appleframeworks',
  modules : [
    'Metal',
    'MetalKit',
    'Cocoa',
    'CoreAudio',
    'AudioToolbox',
    'GameController',
    'QuartzCore',
    'CoreVideo',
  ],
)

# Shader compilation pipeline
subdir('shaders')

# Source modules
subdir('src')

# Tests
subdir('tests')

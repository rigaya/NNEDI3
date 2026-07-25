project('nnedi3', 'cpp',
  version : '1.0.0',
  default_options : [
    'cpp_std=c++17',
    'warning_level=0',
    'buildtype=release',
    'libdir=lib/avisynth'  # インストール先を指定
  ]
)

avisynth_dep = dependency('avisynth', required : true)

if host_machine.system() == 'windows'
    error('Please use msbuild to build the solution.')
endif

cpp_args = [ '-std=c++17',
'-fno-operator-names',
'-Wno-unused-variable',
'-Wno-unused-function',
'-Wno-unused-value',
'-Wno-unused-label',
'-Wno-unused-parameter',
'-Wno-unused-type',
'-Wno-unknown-pragmas',
'-Wno-unused-but-set-variable',
'-Wno-reorder',
'-Wno-sign-conversion'
]
link_args = []

# SIMD 拡張命令セット用のコンパイルフラグ
sse2_args = cpp_args + ['-msse2']
ssse3_args = cpp_args + ['-mssse3']
sse41_args = cpp_args + ['-msse4.1']
avx_args = cpp_args + ['-mavx', '-mpopcnt']
avx2_args = cpp_args + ['-mavx2', '-mfma']
avx512_args = cpp_args + ['-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl', '-mavx2', '-mfma']

subdir('nnedi3')

project('nnedi3-avxvnni-test', 'cpp',
  default_options : ['cpp_std=c++17', 'buildtype=release', 'warning_level=2'])

cpp = meson.get_compiler('cpp')
if cpp.get_argument_syntax() == 'msvc'
  avx2_args = ['/arch:AVX2', '/utf-8']
else
  avx2_args = ['-mavx2', '-mfma']
endif

nnedi3_include = include_directories('../../nnedi3')

avx2_library = static_library('nnedi3-avx2-for-test',
  '../../nnedi3/nnedi3_intrinsic_FMA_x64.cpp',
  include_directories : nnedi3_include,
  cpp_args : avx2_args,
)

avxvnni_library = static_library('nnedi3-avxvnni-for-test',
  '../../nnedi3/nnedi3_intrinsic_AVXVNNI.cpp',
  include_directories : nnedi3_include,
  cpp_args : avx2_args,
)

avxvnni_test = executable('nnedi3-avxvnni-test',
  'avxvnni_test.cpp',
  include_directories : nnedi3_include,
  link_with : [avx2_library, avxvnni_library],
)

test('AVX-VNNI出力一致', avxvnni_test,
  args : ['--correctness'],
  timeout : 120,
)

benchmark('AVX-VNNI予測器', avxvnni_test,
  args : ['--benchmark'],
  timeout : 120,
)

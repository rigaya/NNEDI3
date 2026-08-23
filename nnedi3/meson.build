nnedi3_sources = [
  'nnedi3.cpp',
  'PlanarFrame.cpp',
  'ThreadPool.cpp',
  'ThreadPoolInterface.cpp',
  'compat/rgy_event.cpp'
]

nnedi3_avx2_sources = [
  'nnedi3_intrinsic_FMA_x64.cpp',
  'nnedi3_intrinsic_AVXVNNI.cpp',
  'nnedi3_intrinsic_AVXVNNI.h',
  'nnedi3_intrinsic.h'
]

nnedi3_avx512_sources = [
  'nnedi3_intrinsic_AVX512.cpp',
  'nnedi3_intrinsic_AVX512_extract.cpp',
  'nnedi3_intrinsic_AVX512_pixel_convert.cpp',
  'nnedi3_intrinsic_AVX512_prescreener.cpp',
  'nnedi3_intrinsic_AVX512_process.cpp',
]

nnedi3_main_args = cpp_args
if host_machine.system() == 'linux'
  # Cバックエンドの演算順序を維持する
  nnedi3_main_args += ['-ffp-contract=off']
endif

binary1_symbol_args = []

# バイナリデータファイルのobjcopy処理
if not (host_machine.system() == 'windows')
  objcopy = find_program('objcopy', required: true)
  embed_binary = find_program('embed_binary.sh', required: true)
  binary1_obj = custom_target('binary1_obj',
    input: 'binary1.bin',
    output: 'binary1.o',
    command: [embed_binary, objcopy, '@INPUT@', '@OUTPUT@', '@PRIVATE_DIR@', host_machine.cpu_family()],
    build_by_default: true
  )
  binary1_dep = declare_dependency(
    link_with: static_library('binary1', binary1_obj),
    sources: binary1_obj
  )
  binary1_symbol_args = [
    '-D_binary____nnedi3_binary1_bin_start=nnedi3_binary1_start',
    '-D_binary____nnedi3_binary1_bin_end=nnedi3_binary1_end',
  ]
endif

# x86でのみx86専用のSIMD実装をビルドする
if is_x86
  nnedi3_avx2_lib = static_library('nnedi3_avx2',
    nnedi3_avx2_sources,
    include_directories : include_directories('.', './compat'),
    cpp_args : avx2_args,
    name_prefix : '',
  )
  nnedi3_avx2_dep = declare_dependency(link_with: nnedi3_avx2_lib)

  nnedi3_avx512_lib = static_library('nnedi3_avx512',
    nnedi3_avx512_sources,
    include_directories : include_directories('.', './compat'),
    cpp_args : cpp_args + ['-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl', '-mavx2', '-mfma'],
    name_prefix : '',
  )
  nnedi3_avx512_dep = declare_dependency(link_with: nnedi3_avx512_lib)
else
  nnedi3_avx2_dep = declare_dependency()
  nnedi3_avx512_dep = declare_dependency()
endif

nnedi3_linux_link_args = [
  '-Wl,--no-undefined,-z,noexecstack',
  # AviSynth本体にも同名のThreadPoolがあるため、プラグイン内部参照を自身へ束縛する
  '-Wl,-Bsymbolic-functions',
  '-Wl,--version-script=' + join_paths(meson.current_source_dir(), 'nnedi3_linux.map'),
]
if is_x86
  nnedi3_linux_link_args += ['-Wl,--undefined=nnedi3_avx512_build_marker']
endif

shared_module('libnnedi3',
  nnedi3_sources,
  include_directories : include_directories('.', './compat'),
  dependencies : [avisynth_dep, nnedi3_avx2_dep, nnedi3_avx512_dep] + (not (host_machine.system() == 'windows') ? [binary1_dep] : []),
  cpp_args : nnedi3_main_args + binary1_symbol_args,
  link_args : link_args + (host_machine.system() == 'linux' ? nnedi3_linux_link_args : []),
  link_depends : host_machine.system() == 'linux' ? files('nnedi3_linux.map') : [],
  name_prefix : '',
  install : true,
  install_dir : get_option('libdir')
)

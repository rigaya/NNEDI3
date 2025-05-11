nnedi3_sources = [
  'nnedi3.cpp',
  'PlanarFrame.cpp',
  'ThreadPool.cpp',
  'ThreadPoolInterface.cpp',
  'nnedi3_kernel.cu',
]

nnedi3_avx2_sources = [
  'nnedi3_intrinsic_FMA_x64.cpp'
]

nnedi3_headers = [
  'internal.h',
  'knnedi_version.h',
  'nnedi3.h',
  'nnedi3_kernel.h',
  'PlanarFrame.h',
  'resource.h',
  'ThreadPool.h',
  'ThreadPoolLinux.h',
  'ThreadPoolDef.h',
  'ThreadPoolInterface.h',
  'ThreadPoolInterfaceLinux.h',
]

# バイナリデータファイルのobjcopy処理
if not (host_machine.system() == 'windows')
  objcopy = find_program('objcopy', required: true)
  binary1_obj = custom_target('binary1_obj',
    input: 'binary1.bin',
    output: 'binary1.o',
    command: [objcopy, '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64', '--rename-section', '.data=.rodata,alloc,load,readonly,data,contents', '@INPUT@', '@OUTPUT@'],
    build_by_default: true
  )
  binary1_dep = declare_dependency(
    link_with: static_library('binary1', binary1_obj),
    sources: binary1_obj
  )
endif

# AVX2用のstatic_libraryを作成
nnedi3_avx2_lib = static_library('nnedi3_avx2',
  nnedi3_avx2_sources,
  include_directories : include_directories('../..', '../../KUtil',  '../../common'),
  cpp_args : avx2_args,
  name_prefix : '',
)

nnedi3_avx2_dep = declare_dependency(
  link_with: nnedi3_avx2_lib,
)

shared_module('nnedi3',
  nnedi3_sources,
  include_directories : include_directories('../..', '../../KUtil',  '../../common'),
  dependencies : [avisynth_dep, common_dep, cuda_dep, threads_dep, nnedi3_avx2_dep] + (not (host_machine.system() == 'windows') ? [binary1_dep] : []),
  cpp_args : cpp_args,
  cuda_args : cuda_args,
  link_args : link_args,
  name_prefix : '',
  install : true,
  install_dir : get_option('libdir')
)

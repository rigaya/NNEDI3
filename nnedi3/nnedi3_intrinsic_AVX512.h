#pragma once

#include <cstdint>

#if defined(_WIN32)
#define NNEDI3_AVX512_MARKER_EXPORT __declspec(dllexport)
#else
#define NNEDI3_AVX512_MARKER_EXPORT
#endif

extern "C" {

// AVX-512翻訳単位が最終成果物へ含まれたことを確認するための安全な識別子。
// CPU機能の判定には使用せず、実行時ディスパッチが完成するまではkernelを公開しない。
NNEDI3_AVX512_MARKER_EXPORT std::uint32_t nnedi3_avx512_build_marker() noexcept;

}

#undef NNEDI3_AVX512_MARKER_EXPORT

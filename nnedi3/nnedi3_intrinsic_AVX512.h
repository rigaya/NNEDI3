#pragma once

#include <cstdint>

#if defined(_WIN32)
#define NNEDI3_AVX512_MARKER_EXPORT __declspec(dllexport)
#else
#define NNEDI3_AVX512_MARKER_EXPORT
#endif

extern "C" {

// AVX-512翻訳単位が最終成果物へ含まれたことを確認するための安全な識別子。
// CPU機能の判定や実行時ディスパッチには使用しない。
NNEDI3_AVX512_MARKER_EXPORT std::uint32_t nnedi3_avx512_build_marker() noexcept;

void dotProd_m32_m16_AVX512(const float* data, const float* weights,
    float* vals, int n, int len, const float* istd);
void dotProd_m48_m16_AVX512(const float* data, const float* weights,
    float* vals, int n, int len, const float* istd);
void dotProd_m32_m16_i16_AVX512(const float* data, const float* weights,
    float* vals, int n, int len, const float* istd);
void dotProd_m48_m16_i16_AVX512(const float* data, const float* weights,
    float* vals, int n, int len, const float* istd);

}

#undef NNEDI3_AVX512_MARKER_EXPORT

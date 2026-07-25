#include "nnedi3_intrinsic_AVX512_extract.h"

#include <immintrin.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#define NNEDI3_AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma")))
#else
#define NNEDI3_AVX512_TARGET
#endif

namespace {

NNEDI3_AVX512_TARGET
float horizontalSum16(const __m512 value)
{
    const __m256 sum8 = _mm256_add_ps(
        _mm512_castps512_ps256(value), _mm512_extractf32x8_ps(value, 1));
    __m128 sum4 = _mm_add_ps(
        _mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

NNEDI3_AVX512_TARGET
std::int32_t horizontalSum16xInt32(const __m512i value)
{
    const __m256i sum8 = _mm256_add_epi32(
        _mm512_castsi512_si256(value), _mm512_extracti64x4_epi64(value, 1));
    __m128i sum4 = _mm_add_epi32(
        _mm256_castsi256_si128(sum8), _mm256_extracti128_si256(sum8, 1));
    sum4 = _mm_add_epi32(sum4, _mm_shuffle_epi32(sum4, 0x4e));
    sum4 = _mm_add_epi32(sum4, _mm_shuffle_epi32(sum4, 0xb1));
    return _mm_cvtsi128_si32(sum4);
}

NNEDI3_AVX512_TARGET
std::int64_t horizontalSum8xInt64(const __m512i value)
{
    const __m256i sum4 = _mm256_add_epi64(
        _mm512_castsi512_si256(value), _mm512_extracti64x4_epi64(value, 1));
    __m128i sum2 = _mm_add_epi64(
        _mm256_castsi256_si128(sum4), _mm256_extracti128_si256(sum4, 1));
    sum2 = _mm_add_epi64(sum2, _mm_unpackhi_epi64(sum2, sum2));
    std::int64_t result = 0;
    _mm_storel_epi64(reinterpret_cast<__m128i*>(&result), sum2);
    return result;
}

NNEDI3_AVX512_TARGET
void finishFloatStatistics(const __m512 sum, const __m512 sumsq,
    const int count, float* const mstd)
{
    const __m128 countValue = _mm_set_ss(static_cast<float>(count));
    const __m128 mean = _mm_div_ss(_mm_set_ss(horizontalSum16(sum)), countValue);
    const __m128 averageSquare = _mm_div_ss(
        _mm_set_ss(horizontalSum16(sumsq)), countValue);
    // 分散の積差も1回の丸めに統一する。
    const __m128 variance = _mm_fnmadd_ss(mean, mean, averageSquare);
    mstd[0] = _mm_cvtss_f32(mean);
    if (!(_mm_cvtss_f32(variance) > FLT_EPSILON)) {
        mstd[1] = 0.0f;
        mstd[2] = 0.0f;
    } else {
        const __m128 stdDev = _mm_sqrt_ss(variance);
        mstd[1] = _mm_cvtss_f32(stdDev);
        mstd[2] = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(1.0f), stdDev));
    }
    mstd[3] = 0.0f;
}

NNEDI3_AVX512_TARGET
void finishInt32Statistics(const __m512i sum, const __m512i sumsq,
    const int count, float* const mstd)
{
    const __m128 countValue = _mm_set_ss(static_cast<float>(count));
    const __m128 mean = _mm_div_ss(
        _mm_cvtsi32_ss(_mm_setzero_ps(), horizontalSum16xInt32(sum)), countValue);
    const __m128 averageSquare = _mm_div_ss(
        _mm_cvtsi32_ss(_mm_setzero_ps(), horizontalSum16xInt32(sumsq)), countValue);
    const __m128 variance = _mm_fnmadd_ss(mean, mean, averageSquare);
    mstd[0] = _mm_cvtss_f32(mean);
    if (!(_mm_cvtss_f32(variance) > FLT_EPSILON)) {
        mstd[1] = 0.0f;
        mstd[2] = 0.0f;
    } else {
        const __m128 stdDev = _mm_sqrt_ss(variance);
        mstd[1] = _mm_cvtss_f32(stdDev);
        mstd[2] = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(1.0f), stdDev));
    }
    mstd[3] = 0.0f;
}

void finishWideStatistics(const std::int32_t sum, const std::int64_t sumsq,
    const int count, float* const mstd)
{
    const float scale = static_cast<float>(1.0 / static_cast<double>(count));
    const float mean = static_cast<float>(sum) * scale;
    const double meanSquare = static_cast<double>(mean) * mean;
    const double variance = std::fma(
        static_cast<double>(sumsq), static_cast<double>(scale), -meanSquare);
    mstd[0] = mean;
    if (!(variance > FLT_EPSILON)) {
        mstd[1] = 0.0f;
        mstd[2] = 0.0f;
    } else {
        mstd[1] = static_cast<float>(std::sqrt(variance));
        mstd[2] = 1.0f / mstd[1];
    }
    mstd[3] = 0.0f;
}

NNEDI3_AVX512_TARGET
__mmask16 mask16(const int count)
{
    return static_cast<__mmask16>((UINT32_C(1) << count) - UINT32_C(1));
}

NNEDI3_AVX512_TARGET
__mmask32 mask32(const int count)
{
    return count == 32 ? static_cast<__mmask32>(UINT32_MAX)
                       : static_cast<__mmask32>((UINT32_C(1) << count) - UINT32_C(1));
}

NNEDI3_AVX512_TARGET
void accumulateFloatPair(const __m512 row0, const __m512 row1,
    __m512& sum, __m512& sumsq)
{
    sum = _mm512_add_ps(sum, row0);
    sum = _mm512_add_ps(sum, row1);
    sumsq = _mm512_fmadd_ps(row0, row0, sumsq);
    sumsq = _mm512_fmadd_ps(row1, row1, sumsq);
}

} // 無名名前空間

extern "C" NNEDI3_AVX512_TARGET
void extract_m8_AVX512(const std::uint8_t* const src, const int stride,
    const int xdia, const int ydia, float* const mstd, float* const input)
{
    __m512 sum = _mm512_setzero_ps();
    __m512 sumsq = _mm512_setzero_ps();
    for (int y = 0; y < ydia; y += 2) {
        const std::uint8_t* const row0 = src + static_cast<std::ptrdiff_t>(y) * stride * 2;
        const std::uint8_t* const row1 = row0 + stride * 2;
        float* const output0 = input + y * xdia;
        float* const output1 = output0 + xdia;
        for (int x = 0; x < xdia; x += 16) {
            const int block = xdia - x < 16 ? xdia - x : 16;
            const __mmask16 mask = mask16(block);
            const __m128i bytes0 = _mm_maskz_loadu_epi8(mask, row0 + x);
            const __m128i bytes1 = _mm_maskz_loadu_epi8(mask, row1 + x);
            const __m512 pixels0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes0));
            const __m512 pixels1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes1));
            _mm512_mask_storeu_ps(output0 + x, mask, pixels0);
            _mm512_mask_storeu_ps(output1 + x, mask, pixels1);
            accumulateFloatPair(pixels0, pixels1, sum, sumsq);
        }
    }
    finishFloatStatistics(sum, sumsq, xdia * ydia, mstd);
    _mm256_zeroupper();
}

extern "C" NNEDI3_AVX512_TARGET
void extract_m8_AVX512_16(const std::uint8_t* const src, const int stride,
    const int xdia, const int ydia, float* const mstd, float* const input)
{
    __m512 sum = _mm512_setzero_ps();
    __m512 sumsq = _mm512_setzero_ps();
    for (int y = 0; y < ydia; y += 2) {
        const std::uint8_t* const row0 = src + static_cast<std::ptrdiff_t>(y) * stride * 2;
        const std::uint8_t* const row1 = row0 + stride * 2;
        float* const output0 = input + y * xdia;
        float* const output1 = output0 + xdia;
        for (int x = 0; x < xdia; x += 16) {
            const int block = xdia - x < 16 ? xdia - x : 16;
            const __mmask16 mask = mask16(block);
            const __m256i words0 = _mm256_maskz_loadu_epi16(mask,
                reinterpret_cast<const void*>(row0 + x * 2));
            const __m256i words1 = _mm256_maskz_loadu_epi16(mask,
                reinterpret_cast<const void*>(row1 + x * 2));
            const __m512 pixels0 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(words0));
            const __m512 pixels1 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(words1));
            _mm512_mask_storeu_ps(output0 + x, mask, pixels0);
            _mm512_mask_storeu_ps(output1 + x, mask, pixels1);
            accumulateFloatPair(pixels0, pixels1, sum, sumsq);
        }
    }
    finishFloatStatistics(sum, sumsq, xdia * ydia, mstd);
    _mm256_zeroupper();
}

extern "C" NNEDI3_AVX512_TARGET
void extract_m8_AVX512_32(const std::uint8_t* const src, const int stride,
    const int xdia, const int ydia, float* const mstd, float* const input)
{
    __m512 sum = _mm512_setzero_ps();
    __m512 sumsq = _mm512_setzero_ps();
    for (int y = 0; y < ydia; y += 2) {
        const float* const row0 = reinterpret_cast<const float*>(
            src + static_cast<std::ptrdiff_t>(y) * stride * 2);
        const float* const row1 = reinterpret_cast<const float*>(
            reinterpret_cast<const std::uint8_t*>(row0) + stride * 2);
        float* const output0 = input + y * xdia;
        float* const output1 = output0 + xdia;
        for (int x = 0; x < xdia; x += 16) {
            const int block = xdia - x < 16 ? xdia - x : 16;
            const __mmask16 mask = mask16(block);
            const __m512 pixels0 = _mm512_maskz_loadu_ps(mask, row0 + x);
            const __m512 pixels1 = _mm512_maskz_loadu_ps(mask, row1 + x);
            _mm512_mask_storeu_ps(output0 + x, mask, pixels0);
            _mm512_mask_storeu_ps(output1 + x, mask, pixels1);
            accumulateFloatPair(pixels0, pixels1, sum, sumsq);
        }
    }
    finishFloatStatistics(sum, sumsq, xdia * ydia, mstd);
    _mm256_zeroupper();
}

extern "C" NNEDI3_AVX512_TARGET
void extract_m8_i16_AVX512(const std::uint8_t* const src, const int stride,
    const int xdia, const int ydia, float* const mstd, float* const inputRaw)
{
    auto* const input = reinterpret_cast<std::uint16_t*>(inputRaw);
    const __m512i ones = _mm512_set1_epi16(1);
    __m512i sum = _mm512_setzero_si512();
    __m512i sumsq = _mm512_setzero_si512();
    for (int y = 0; y < ydia; ++y) {
        const std::uint8_t* const row = src + static_cast<std::ptrdiff_t>(y) * stride * 2;
        std::uint16_t* const output = input + y * xdia;
        for (int x = 0; x < xdia; x += 32) {
            const int block = xdia - x < 32 ? xdia - x : 32;
            const __mmask32 mask = mask32(block);
            const __m256i bytes = _mm256_maskz_loadu_epi8(mask, row + x);
            const __m512i words = _mm512_cvtepu8_epi16(bytes);
            _mm512_mask_storeu_epi16(output + x, mask, words);
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(words, ones));
            sumsq = _mm512_add_epi32(sumsq, _mm512_madd_epi16(words, words));
        }
    }
    finishInt32Statistics(sum, sumsq, xdia * ydia, mstd);
    _mm256_zeroupper();
}

extern "C" NNEDI3_AVX512_TARGET
void extract_m8_i16_AVX512_16(const std::uint8_t* const src, const int stride,
    const int xdia, const int ydia, float* const mstd, float* const inputRaw)
{
    auto* const input = reinterpret_cast<std::uint16_t*>(inputRaw);
    const __m512i ones = _mm512_set1_epi16(1);
    __m512i sum = _mm512_setzero_si512();
    __m512i sumsqLow = _mm512_setzero_si512();
    __m512i sumsqHigh = _mm512_setzero_si512();
    for (int y = 0; y < ydia; ++y) {
        const std::uint8_t* const row = src + static_cast<std::ptrdiff_t>(y) * stride * 2;
        std::uint16_t* const output = input + y * xdia;
        for (int x = 0; x < xdia; x += 32) {
            const int block = xdia - x < 32 ? xdia - x : 32;
            const __mmask32 mask = mask32(block);
            const __m512i words = _mm512_maskz_loadu_epi16(mask,
                reinterpret_cast<const void*>(row + x * 2));
            _mm512_mask_storeu_epi16(output + x, mask, words);
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(words, ones));
            const __m512i pairSquares = _mm512_madd_epi16(words, words);
            sumsqLow = _mm512_add_epi64(sumsqLow,
                _mm512_cvtepi32_epi64(_mm512_castsi512_si256(pairSquares)));
            sumsqHigh = _mm512_add_epi64(sumsqHigh,
                _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(pairSquares, 1)));
        }
    }
    const std::int32_t total = horizontalSum16xInt32(sum);
    const std::int64_t totalSquare = horizontalSum8xInt64(
        _mm512_add_epi64(sumsqLow, sumsqHigh));
    finishWideStatistics(total, totalSquare, xdia * ydia, mstd);
    _mm256_zeroupper();
}

extern "C" NNEDI3_AVX512_TARGET
void extract_m8_i16_AVX512_16_10(const std::uint8_t* const src, const int stride,
    const int xdia, const int ydia, float* const mstd, float* const inputRaw)
{
    auto* const input = reinterpret_cast<std::uint16_t*>(inputRaw);
    const __m512i ones = _mm512_set1_epi16(1);
    __m512i sum = _mm512_setzero_si512();
    __m512i sumsq = _mm512_setzero_si512();
    for (int y = 0; y < ydia; ++y) {
        const std::uint8_t* const row = src + static_cast<std::ptrdiff_t>(y) * stride * 2;
        std::uint16_t* const output = input + y * xdia;
        for (int x = 0; x < xdia; x += 32) {
            const int block = xdia - x < 32 ? xdia - x : 32;
            const __mmask32 mask = mask32(block);
            const __m512i words = _mm512_maskz_loadu_epi16(mask,
                reinterpret_cast<const void*>(row + x * 2));
            _mm512_mask_storeu_epi16(output + x, mask, words);
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(words, ones));
            sumsq = _mm512_add_epi32(sumsq, _mm512_madd_epi16(words, words));
        }
    }
    finishInt32Statistics(sum, sumsq, xdia * ydia, mstd);
    _mm256_zeroupper();
}

#undef NNEDI3_AVX512_TARGET

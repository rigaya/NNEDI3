#include "nnedi3_intrinsic_AVX512.h"

#include <immintrin.h>

#include <cstddef>

namespace {

float horizontalSum16(const __m512 value)
{
    // 縮約順をコンパイラ任せにせず、両OSで同じ加算木を使用する。
    const __m256 sum8 = _mm256_add_ps(
        _mm512_castps512_ps256(value), _mm512_extractf32x8_ps(value, 1));
    __m128 sum4 = _mm_add_ps(
        _mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

std::int32_t horizontalSum16xInt32(const __m512i value)
{
    // VPADDDと同じ32bit wrap加算だけで縮約し、AVX2版の整数値を保つ。
    const __m256i sum8 = _mm256_add_epi32(
        _mm512_castsi512_si256(value), _mm512_extracti64x4_epi64(value, 1));
    __m128i sum4 = _mm_add_epi32(
        _mm256_castsi256_si128(sum8), _mm256_extracti128_si256(sum8, 1));
    sum4 = _mm_add_epi32(sum4, _mm_shuffle_epi32(sum4, 0x4e));
    sum4 = _mm_add_epi32(sum4, _mm_shuffle_epi32(sum4, 0xb1));
    return _mm_cvtsi128_si32(sum4);
}

void dotProdFloatAVX512(const float* data, const float* weights,
    float* vals, const int n, const int len, const float* istd)
{
    const float* const bias = weights + static_cast<std::size_t>(n) * len;
    const __m128 inverseStdDev = _mm_set1_ps(*istd);

    for (int neuron = 0; neuron < n; neuron += 4) {
        const float* const groupWeights = weights
            + static_cast<std::size_t>(neuron) * len;
        __m512 sums0 = _mm512_setzero_ps();
        __m512 sums1 = _mm512_setzero_ps();
        __m512 sums2 = _mm512_setzero_ps();
        __m512 sums3 = _mm512_setzero_ps();

        for (int input = 0; input < len; input += 16) {
            const __m512 values = _mm512_loadu_ps(data + input);
            const float* const tile = groupWeights
                + static_cast<std::size_t>(input) * 4;
            sums0 = _mm512_fmadd_ps(values, _mm512_loadu_ps(tile), sums0);
            sums1 = _mm512_fmadd_ps(values, _mm512_loadu_ps(tile + 16), sums1);
            sums2 = _mm512_fmadd_ps(values, _mm512_loadu_ps(tile + 32), sums2);
            sums3 = _mm512_fmadd_ps(values, _mm512_loadu_ps(tile + 48), sums3);
        }

        const __m128 dotProducts = _mm_setr_ps(
            horizontalSum16(sums0), horizontalSum16(sums1),
            horizontalSum16(sums2), horizontalSum16(sums3));
        const __m128 result = _mm_fmadd_ps(
            dotProducts, inverseStdDev, _mm_loadu_ps(bias + neuron));
        _mm_storeu_ps(vals + neuron, result);
    }
}

void dotProdInt16AVX512(const float* dataRaw, const float* weightsRaw,
    float* vals, const int n, const int len, const float* istd)
{
    const auto* const data = reinterpret_cast<const std::int16_t*>(dataRaw);
    const auto* const weights = reinterpret_cast<const std::int16_t*>(weightsRaw);
    const auto* const scaleBias = reinterpret_cast<const float*>(
        weights + static_cast<std::size_t>(n) * len);
    const __m128 inverseStdDev = _mm_set1_ps(*istd);
    const int fullLength = len & ~31;
    const int tailLength = len - fullLength;

    for (int neuron = 0; neuron < n; neuron += 4) {
        const std::int16_t* const groupWeights = weights
            + static_cast<std::size_t>(neuron) * len;
        __m512i sums0 = _mm512_setzero_si512();
        __m512i sums1 = _mm512_setzero_si512();
        __m512i sums2 = _mm512_setzero_si512();
        __m512i sums3 = _mm512_setzero_si512();

        for (int input = 0; input < fullLength; input += 32) {
            const __m512i values = _mm512_loadu_si512(data + input);
            const std::int16_t* const tile = groupWeights
                + static_cast<std::size_t>(input) * 4;
            sums0 = _mm512_add_epi32(sums0,
                _mm512_madd_epi16(values, _mm512_loadu_si512(tile)));
            sums1 = _mm512_add_epi32(sums1,
                _mm512_madd_epi16(values, _mm512_loadu_si512(tile + 32)));
            sums2 = _mm512_add_epi32(sums2,
                _mm512_madd_epi16(values, _mm512_loadu_si512(tile + 64)));
            sums3 = _mm512_add_epi32(sums3,
                _mm512_madd_epi16(values, _mm512_loadu_si512(tile + 96)));
        }

        if (tailLength != 0) {
            const __mmask32 tailMask = static_cast<__mmask32>(
                (UINT32_C(1) << tailLength) - UINT32_C(1));
            const __m512i values = _mm512_maskz_loadu_epi16(
                tailMask, data + fullLength);
            const std::int16_t* const tile = groupWeights
                + static_cast<std::size_t>(fullLength) * 4;
            sums0 = _mm512_add_epi32(sums0, _mm512_madd_epi16(values,
                _mm512_maskz_loadu_epi16(tailMask, tile)));
            sums1 = _mm512_add_epi32(sums1, _mm512_madd_epi16(values,
                _mm512_maskz_loadu_epi16(tailMask, tile + tailLength)));
            sums2 = _mm512_add_epi32(sums2, _mm512_madd_epi16(values,
                _mm512_maskz_loadu_epi16(tailMask, tile + 2 * tailLength)));
            sums3 = _mm512_add_epi32(sums3, _mm512_madd_epi16(values,
                _mm512_maskz_loadu_epi16(tailMask, tile + 3 * tailLength)));
        }

        const __m128i integerSums = _mm_setr_epi32(
            horizontalSum16xInt32(sums0), horizontalSum16xInt32(sums1),
            horizontalSum16xInt32(sums2), horizontalSum16xInt32(sums3));
        const __m128 converted = _mm_cvtepi32_ps(integerSums);
        const float* const groupScaleBias
            = scaleBias + static_cast<std::size_t>(neuron / 4) * 8;
        // scale乗算の丸め後、istd乗算とbias加算をFMAへ融合する。
        const __m128 scaled = _mm_mul_ps(converted, _mm_loadu_ps(groupScaleBias));
        const __m128 result = _mm_fmadd_ps(
            scaled, inverseStdDev, _mm_loadu_ps(groupScaleBias + 4));
        _mm_storeu_ps(vals + neuron, result);
    }
}

} // 無名名前空間

extern "C" std::uint32_t nnedi3_avx512_build_marker() noexcept
{
    // 非対応CPUでも安全に参照できるよう、このmarkerにはAVX-512命令を含めない。
    return UINT32_C(0x41565835);
}

extern "C" void dotProd_m32_m16_AVX512(const float* data, const float* weights,
    float* vals, const int n, const int len, const float* istd)
{
    dotProdFloatAVX512(data, weights, vals, n, len, istd);
}

extern "C" void dotProd_m48_m16_AVX512(const float* data, const float* weights,
    float* vals, const int n, const int len, const float* istd)
{
    dotProdFloatAVX512(data, weights, vals, n, len, istd);
}

extern "C" void dotProd_m32_m16_i16_AVX512(const float* data, const float* weights,
    float* vals, const int n, const int len, const float* istd)
{
    dotProdInt16AVX512(data, weights, vals, n, len, istd);
}

extern "C" void dotProd_m48_m16_i16_AVX512(const float* data, const float* weights,
    float* vals, const int n, const int len, const float* istd)
{
    dotProdInt16AVX512(data, weights, vals, n, len, istd);
}

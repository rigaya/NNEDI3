#include "nnedi3_intrinsic_AVX512.h"

#include <immintrin.h>

#include <cstddef>

#if defined(__GNUC__) || defined(__clang__)
#define NNEDI3_AVX512VNNI_TARGET __attribute__((target("avx512vnni")))
#else
#define NNEDI3_AVX512VNNI_TARGET
#endif

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

__m128i horizontalSum4x16Int32(const __m512i sums0, const __m512i sums1,
    const __m512i sums2, const __m512i sums3)
{
    __m256i sum0 = _mm256_add_epi32(
        _mm512_castsi512_si256(sums0), _mm512_extracti64x4_epi64(sums0, 1));
    __m256i sum1 = _mm256_add_epi32(
        _mm512_castsi512_si256(sums1), _mm512_extracti64x4_epi64(sums1, 1));
    __m256i sum2 = _mm256_add_epi32(
        _mm512_castsi512_si256(sums2), _mm512_extracti64x4_epi64(sums2, 1));
    __m256i sum3 = _mm256_add_epi32(
        _mm512_castsi512_si256(sums3), _mm512_extracti64x4_epi64(sums3, 1));
    const __m256i high01 = _mm256_unpackhi_epi64(sum0, sum1);
    const __m256i high23 = _mm256_unpackhi_epi64(sum2, sum3);
    sum0 = _mm256_add_epi32(_mm256_unpacklo_epi64(sum0, sum1), high01);
    sum2 = _mm256_add_epi32(_mm256_unpacklo_epi64(sum2, sum3), high23);
    const __m128i pair01 = _mm_add_epi32(
        _mm256_castsi256_si128(sum0), _mm256_extracti128_si256(sum0, 1));
    const __m128i pair23 = _mm_add_epi32(
        _mm256_castsi256_si128(sum2), _mm256_extracti128_si256(sum2, 1));
    return _mm_add_epi32(
        _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(pair01), _mm_castsi128_ps(pair23), 0x88)),
        _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(pair01), _mm_castsi128_ps(pair23), 0xdd)));
}

void dotProdInt16Len48AVX512(const std::int16_t* data,
    const std::int16_t* weights, float* vals, const int n, const float istd)
{
    const __m512i data0 = _mm512_loadu_si512(data);
    const __m256i data1 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(data + 32));
    const float* const scaleBias = reinterpret_cast<const float*>(
        weights + static_cast<std::size_t>(n) * 48);
    const __m128 inverseStdDev = _mm_set1_ps(istd);

    for (int neuron = 0; neuron < n; neuron += 4) {
        const std::int16_t* const group = weights
            + static_cast<std::size_t>(neuron) * 48;
        __m512i sums0 = _mm512_madd_epi16(
            data0, _mm512_loadu_si512(group));
        __m512i sums1 = _mm512_madd_epi16(
            data0, _mm512_loadu_si512(group + 32));
        __m512i sums2 = _mm512_madd_epi16(
            data0, _mm512_loadu_si512(group + 64));
        __m512i sums3 = _mm512_madd_epi16(
            data0, _mm512_loadu_si512(group + 96));
        sums0 = _mm512_add_epi32(sums0, _mm512_zextsi256_si512(
            _mm256_madd_epi16(data1, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group + 128)))));
        sums1 = _mm512_add_epi32(sums1, _mm512_zextsi256_si512(
            _mm256_madd_epi16(data1, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group + 144)))));
        sums2 = _mm512_add_epi32(sums2, _mm512_zextsi256_si512(
            _mm256_madd_epi16(data1, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group + 160)))));
        sums3 = _mm512_add_epi32(sums3, _mm512_zextsi256_si512(
            _mm256_madd_epi16(data1, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group + 176)))));

        const __m128 converted = _mm_cvtepi32_ps(
            horizontalSum4x16Int32(sums0, sums1, sums2, sums3));
        const float* const groupScaleBias = scaleBias
            + static_cast<std::size_t>(neuron / 4) * 8;
        const __m128 scaled = _mm_mul_ps(
            converted, _mm_loadu_ps(groupScaleBias));
        _mm_storeu_ps(vals + neuron, _mm_fmadd_ps(
            scaled, inverseStdDev, _mm_loadu_ps(groupScaleBias + 4)));
    }
}

void dotProdInt16Len32AVX512(const std::int16_t* data,
    const std::int16_t* weights, float* vals, const int n, const float istd)
{
    const __m512i values = _mm512_loadu_si512(data);
    const float* const scaleBias = reinterpret_cast<const float*>(
        weights + static_cast<std::size_t>(n) * 32);
    const __m128 inverseStdDev = _mm_set1_ps(istd);

    for (int neuron = 0; neuron < n; neuron += 8) {
        const std::int16_t* const group0 = weights
            + static_cast<std::size_t>(neuron) * 32;
        const std::int16_t* const group1 = group0 + 4 * 32;
        const __m512i sums0 = _mm512_madd_epi16(values, _mm512_loadu_si512(group0));
        const __m512i sums1 = _mm512_madd_epi16(values, _mm512_loadu_si512(group0 + 32));
        const __m512i sums2 = _mm512_madd_epi16(values, _mm512_loadu_si512(group0 + 64));
        const __m512i sums3 = _mm512_madd_epi16(values, _mm512_loadu_si512(group0 + 96));
        const __m512i sums4 = _mm512_madd_epi16(values, _mm512_loadu_si512(group1));
        const __m512i sums5 = _mm512_madd_epi16(values, _mm512_loadu_si512(group1 + 32));
        const __m512i sums6 = _mm512_madd_epi16(values, _mm512_loadu_si512(group1 + 64));
        const __m512i sums7 = _mm512_madd_epi16(values, _mm512_loadu_si512(group1 + 96));
        const __m128i integerSums0 = horizontalSum4x16Int32(
            sums0, sums1, sums2, sums3);
        const __m128i integerSums1 = horizontalSum4x16Int32(
            sums4, sums5, sums6, sums7);
        const float* const groupScaleBias = scaleBias
            + static_cast<std::size_t>(neuron / 4) * 8;
        const __m128 scaled0 = _mm_mul_ps(
            _mm_cvtepi32_ps(integerSums0), _mm_loadu_ps(groupScaleBias));
        const __m128 scaled1 = _mm_mul_ps(
            _mm_cvtepi32_ps(integerSums1), _mm_loadu_ps(groupScaleBias + 8));
        _mm_storeu_ps(vals + neuron, _mm_fmadd_ps(
            scaled0, inverseStdDev, _mm_loadu_ps(groupScaleBias + 4)));
        _mm_storeu_ps(vals + neuron + 4, _mm_fmadd_ps(
            scaled1, inverseStdDev, _mm_loadu_ps(groupScaleBias + 12)));
    }
}

void dotProdInt16Len128AVX512(const std::int16_t* data,
    const std::int16_t* weights, float* vals, const int n, const float istd)
{
    const __m512i data0 = _mm512_loadu_si512(data);
    const __m512i data1 = _mm512_loadu_si512(data + 32);
    const __m512i data2 = _mm512_loadu_si512(data + 64);
    const __m512i data3 = _mm512_loadu_si512(data + 96);
    const float* const scaleBias = reinterpret_cast<const float*>(
        weights + static_cast<std::size_t>(n) * 128);
    const __m128 inverseStdDev = _mm_set1_ps(istd);

    for (int neuron = 0; neuron < n; neuron += 8) {
        const std::int16_t* const group0 = weights
            + static_cast<std::size_t>(neuron) * 128;
        const std::int16_t* const group1 = group0 + 4 * 128;
        __m512i sums0 = _mm512_setzero_si512();
        __m512i sums1 = _mm512_setzero_si512();
        __m512i sums2 = _mm512_setzero_si512();
        __m512i sums3 = _mm512_setzero_si512();
        __m512i sums4 = _mm512_setzero_si512();
        __m512i sums5 = _mm512_setzero_si512();
        __m512i sums6 = _mm512_setzero_si512();
        __m512i sums7 = _mm512_setzero_si512();

#define NNEDI3_MADD8_AVX512(values, offset) \
        sums0 = _mm512_add_epi32(sums0, _mm512_madd_epi16(values, _mm512_loadu_si512(group0 + (offset) * 4))); \
        sums1 = _mm512_add_epi32(sums1, _mm512_madd_epi16(values, _mm512_loadu_si512(group0 + (offset) * 4 + 32))); \
        sums2 = _mm512_add_epi32(sums2, _mm512_madd_epi16(values, _mm512_loadu_si512(group0 + (offset) * 4 + 64))); \
        sums3 = _mm512_add_epi32(sums3, _mm512_madd_epi16(values, _mm512_loadu_si512(group0 + (offset) * 4 + 96))); \
        sums4 = _mm512_add_epi32(sums4, _mm512_madd_epi16(values, _mm512_loadu_si512(group1 + (offset) * 4))); \
        sums5 = _mm512_add_epi32(sums5, _mm512_madd_epi16(values, _mm512_loadu_si512(group1 + (offset) * 4 + 32))); \
        sums6 = _mm512_add_epi32(sums6, _mm512_madd_epi16(values, _mm512_loadu_si512(group1 + (offset) * 4 + 64))); \
        sums7 = _mm512_add_epi32(sums7, _mm512_madd_epi16(values, _mm512_loadu_si512(group1 + (offset) * 4 + 96)))

        NNEDI3_MADD8_AVX512(data0, 0);
        NNEDI3_MADD8_AVX512(data1, 32);
        NNEDI3_MADD8_AVX512(data2, 64);
        NNEDI3_MADD8_AVX512(data3, 96);
#undef NNEDI3_MADD8_AVX512

        const __m128i integerSums0 = horizontalSum4x16Int32(
            sums0, sums1, sums2, sums3);
        const __m128i integerSums1 = horizontalSum4x16Int32(
            sums4, sums5, sums6, sums7);
        const float* const groupScaleBias = scaleBias
            + static_cast<std::size_t>(neuron / 4) * 8;
        const __m128 scaled0 = _mm_mul_ps(
            _mm_cvtepi32_ps(integerSums0), _mm_loadu_ps(groupScaleBias));
        const __m128 scaled1 = _mm_mul_ps(
            _mm_cvtepi32_ps(integerSums1), _mm_loadu_ps(groupScaleBias + 8));
        _mm_storeu_ps(vals + neuron, _mm_fmadd_ps(
            scaled0, inverseStdDev, _mm_loadu_ps(groupScaleBias + 4)));
        _mm_storeu_ps(vals + neuron + 4, _mm_fmadd_ps(
            scaled1, inverseStdDev, _mm_loadu_ps(groupScaleBias + 12)));
    }
}

NNEDI3_AVX512VNNI_TARGET
void dotProdInt16Len128AVX512VNNI(const std::int16_t* data,
    const std::int16_t* weights, float* vals, const int n, const float istd)
{
    const __m512i data0 = _mm512_loadu_si512(data);
    const __m512i data1 = _mm512_loadu_si512(data + 32);
    const __m512i data2 = _mm512_loadu_si512(data + 64);
    const __m512i data3 = _mm512_loadu_si512(data + 96);
    const float* const scaleBias = reinterpret_cast<const float*>(
        weights + static_cast<std::size_t>(n) * 128);
    const __m128 inverseStdDev = _mm_set1_ps(istd);

    for (int neuron = 0; neuron < n; neuron += 4) {
        const std::int16_t* const group = weights
            + static_cast<std::size_t>(neuron) * 128;
        __m512i sums0 = _mm512_setzero_si512();
        __m512i sums1 = _mm512_setzero_si512();
        __m512i sums2 = _mm512_setzero_si512();
        __m512i sums3 = _mm512_setzero_si512();

        sums0 = _mm512_dpwssd_epi32(sums0, data0, _mm512_loadu_si512(group));
        sums1 = _mm512_dpwssd_epi32(sums1, data0, _mm512_loadu_si512(group + 32));
        sums2 = _mm512_dpwssd_epi32(sums2, data0, _mm512_loadu_si512(group + 64));
        sums3 = _mm512_dpwssd_epi32(sums3, data0, _mm512_loadu_si512(group + 96));
        sums0 = _mm512_dpwssd_epi32(sums0, data1, _mm512_loadu_si512(group + 128));
        sums1 = _mm512_dpwssd_epi32(sums1, data1, _mm512_loadu_si512(group + 160));
        sums2 = _mm512_dpwssd_epi32(sums2, data1, _mm512_loadu_si512(group + 192));
        sums3 = _mm512_dpwssd_epi32(sums3, data1, _mm512_loadu_si512(group + 224));
        sums0 = _mm512_dpwssd_epi32(sums0, data2, _mm512_loadu_si512(group + 256));
        sums1 = _mm512_dpwssd_epi32(sums1, data2, _mm512_loadu_si512(group + 288));
        sums2 = _mm512_dpwssd_epi32(sums2, data2, _mm512_loadu_si512(group + 320));
        sums3 = _mm512_dpwssd_epi32(sums3, data2, _mm512_loadu_si512(group + 352));
        sums0 = _mm512_dpwssd_epi32(sums0, data3, _mm512_loadu_si512(group + 384));
        sums1 = _mm512_dpwssd_epi32(sums1, data3, _mm512_loadu_si512(group + 416));
        sums2 = _mm512_dpwssd_epi32(sums2, data3, _mm512_loadu_si512(group + 448));
        sums3 = _mm512_dpwssd_epi32(sums3, data3, _mm512_loadu_si512(group + 480));

        const __m128i integerSums = horizontalSum4x16Int32(
            sums0, sums1, sums2, sums3);
        const float* const groupScaleBias = scaleBias
            + static_cast<std::size_t>(neuron / 4) * 8;
        const __m128 scaled = _mm_mul_ps(
            _mm_cvtepi32_ps(integerSums), _mm_loadu_ps(groupScaleBias));
        _mm_storeu_ps(vals + neuron, _mm_fmadd_ps(
            scaled, inverseStdDev, _mm_loadu_ps(groupScaleBias + 4)));
    }
}

NNEDI3_AVX512VNNI_TARGET
void dotProdInt16AVX512VNNI(const float* dataRaw, const float* weightsRaw,
    float* vals, const int n, const int len, const float* istd)
{
    const auto* const data = reinterpret_cast<const std::int16_t*>(dataRaw);
    const auto* const weights = reinterpret_cast<const std::int16_t*>(weightsRaw);
    // 48要素では後半16要素をYMMで処理する非VNNI版の方が速い。
    if (len == 48) {
        dotProdInt16Len48AVX512(data, weights, vals, n, *istd);
        return;
    }
    if (len == 32) {
        dotProdInt16Len32AVX512(data, weights, vals, n, *istd);
        return;
    }
    if (len == 128) {
        dotProdInt16Len128AVX512(data, weights, vals, n, *istd);
        return;
    }
    if (len == 128) {
        dotProdInt16Len128AVX512VNNI(data, weights, vals, n, *istd);
        return;
    }
    const auto* const scaleBias = reinterpret_cast<const float*>(
        weights + static_cast<std::size_t>(n) * len);
    const __m128 inverseStdDev = _mm_set1_ps(*istd);
    const int fullLength = len & ~31;
    const int tailLength = len - fullLength;

    for (int neuron = 0; neuron < n; neuron += 8) {
        const std::int16_t* const group0 = weights
            + static_cast<std::size_t>(neuron) * len;
        const std::int16_t* const group1 = group0
            + static_cast<std::size_t>(4) * len;
        __m512i sums0 = _mm512_setzero_si512();
        __m512i sums1 = _mm512_setzero_si512();
        __m512i sums2 = _mm512_setzero_si512();
        __m512i sums3 = _mm512_setzero_si512();
        __m512i sums4 = _mm512_setzero_si512();
        __m512i sums5 = _mm512_setzero_si512();
        __m512i sums6 = _mm512_setzero_si512();
        __m512i sums7 = _mm512_setzero_si512();

        for (int input = 0; input < fullLength; input += 32) {
            const __m512i values = _mm512_loadu_si512(data + input);
            const std::int16_t* const tile0 = group0
                + static_cast<std::size_t>(input) * 4;
            const std::int16_t* const tile1 = group1
                + static_cast<std::size_t>(input) * 4;
            sums0 = _mm512_dpwssd_epi32(sums0, values, _mm512_loadu_si512(tile0));
            sums1 = _mm512_dpwssd_epi32(sums1, values, _mm512_loadu_si512(tile0 + 32));
            sums2 = _mm512_dpwssd_epi32(sums2, values, _mm512_loadu_si512(tile0 + 64));
            sums3 = _mm512_dpwssd_epi32(sums3, values, _mm512_loadu_si512(tile0 + 96));
            sums4 = _mm512_dpwssd_epi32(sums4, values, _mm512_loadu_si512(tile1));
            sums5 = _mm512_dpwssd_epi32(sums5, values, _mm512_loadu_si512(tile1 + 32));
            sums6 = _mm512_dpwssd_epi32(sums6, values, _mm512_loadu_si512(tile1 + 64));
            sums7 = _mm512_dpwssd_epi32(sums7, values, _mm512_loadu_si512(tile1 + 96));
        }

        if (tailLength != 0) {
            const __mmask32 tailMask = static_cast<__mmask32>(
                (UINT32_C(1) << tailLength) - UINT32_C(1));
            const __m512i values = _mm512_maskz_loadu_epi16(
                tailMask, data + fullLength);
            const std::int16_t* const tile0 = group0
                + static_cast<std::size_t>(fullLength) * 4;
            const std::int16_t* const tile1 = group1
                + static_cast<std::size_t>(fullLength) * 4;
            sums0 = _mm512_dpwssd_epi32(sums0, values,
                _mm512_maskz_loadu_epi16(tailMask, tile0));
            sums1 = _mm512_dpwssd_epi32(sums1, values,
                _mm512_maskz_loadu_epi16(tailMask, tile0 + tailLength));
            sums2 = _mm512_dpwssd_epi32(sums2, values,
                _mm512_maskz_loadu_epi16(tailMask, tile0 + 2 * tailLength));
            sums3 = _mm512_dpwssd_epi32(sums3, values,
                _mm512_maskz_loadu_epi16(tailMask, tile0 + 3 * tailLength));
            sums4 = _mm512_dpwssd_epi32(sums4, values,
                _mm512_maskz_loadu_epi16(tailMask, tile1));
            sums5 = _mm512_dpwssd_epi32(sums5, values,
                _mm512_maskz_loadu_epi16(tailMask, tile1 + tailLength));
            sums6 = _mm512_dpwssd_epi32(sums6, values,
                _mm512_maskz_loadu_epi16(tailMask, tile1 + 2 * tailLength));
            sums7 = _mm512_dpwssd_epi32(sums7, values,
                _mm512_maskz_loadu_epi16(tailMask, tile1 + 3 * tailLength));
        }

        const __m128i integerSums0 = horizontalSum4x16Int32(
            sums0, sums1, sums2, sums3);
        const __m128i integerSums1 = horizontalSum4x16Int32(
            sums4, sums5, sums6, sums7);
        const float* const groupScaleBias = scaleBias
            + static_cast<std::size_t>(neuron / 4) * 8;
        const __m128 scaled0 = _mm_mul_ps(
            _mm_cvtepi32_ps(integerSums0), _mm_loadu_ps(groupScaleBias));
        const __m128 scaled1 = _mm_mul_ps(
            _mm_cvtepi32_ps(integerSums1), _mm_loadu_ps(groupScaleBias + 8));
        _mm_storeu_ps(vals + neuron, _mm_fmadd_ps(
            scaled0, inverseStdDev, _mm_loadu_ps(groupScaleBias + 4)));
        _mm_storeu_ps(vals + neuron + 4, _mm_fmadd_ps(
            scaled1, inverseStdDev, _mm_loadu_ps(groupScaleBias + 12)));
    }
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
    if (len == 48) {
        dotProdInt16Len48AVX512(data, weights, vals, n, *istd);
        return;
    }
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

        const __m128i integerSums = horizontalSum4x16Int32(
            sums0, sums1, sums2, sums3);
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

__m512 clampExpInput(const __m512 value)
{
    return _mm512_max_ps(
        _mm512_min_ps(value, _mm512_set1_ps(80.0f)),
        _mm512_set1_ps(-80.0f));
}

void expApprox0AVX512(float* values, const int n)
{
    const __m512 multiplier = _mm512_set1_ps(12102203.161561486f);
    const __m512 bias = _mm512_set1_ps(1064866805.0f);
    for (int i = 0; i < n; i += 16) {
        const __m512 value = clampExpInput(_mm512_loadu_ps(values + i));
        const __m512 encoded = _mm512_fmadd_ps(value, multiplier, bias);
        _mm512_storeu_ps(values + i,
            _mm512_castsi512_ps(_mm512_cvtps_epi32(encoded)));
    }
}

void expApprox1AVX512(float* values, const int n)
{
    const __m512 scale = _mm512_set1_ps(1.4426950409f);
    const __m512 magicBias = _mm512_set1_ps(12582912.0f);
    const __m512 c0 = _mm512_set1_ps(1.00035f);
    const __m512 c1 = _mm512_set1_ps(0.701277797f);
    const __m512 c2 = _mm512_set1_ps(0.237348593f);
    for (int i = 0; i < n; i += 16) {
        const __m512 value = clampExpInput(_mm512_loadu_ps(values + i));
        const __m512 biased = _mm512_fmadd_ps(value, scale, magicBias);
        const __m512 exponentValue = _mm512_sub_ps(biased, magicBias);
        const __m512 fraction = _mm512_fmsub_ps(value, scale, exponentValue);
        const __m512 square = _mm512_mul_ps(fraction, fraction);
        const __m512 linear = _mm512_fmadd_ps(c1, fraction, c0);
        const __m512 polynomial = _mm512_fmadd_ps(c2, square, linear);
        const __m512i exponentBits = _mm512_slli_epi32(
            _mm512_castps_si512(biased), 23);
        _mm512_storeu_ps(values + i, _mm512_castsi512_ps(
            _mm512_add_epi32(_mm512_castps_si512(polynomial), exponentBits)));
    }
}

void expApprox2AVX512(float* values, const int n)
{
    const __m512 reciprocalLn2 = _mm512_set1_ps(1.442695041f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 c2 = _mm512_set1_ps(1.428606820e-6f);
    const __m512 c1 = _mm512_set1_ps(6.931457520e-1f);
    const __m512 q0 = _mm512_set1_ps(3.001985051e-6f);
    const __m512 p0 = _mm512_set1_ps(1.261771931e-4f);
    const __m512 q1 = _mm512_set1_ps(2.524483403e-3f);
    const __m512 p1 = _mm512_set1_ps(3.029944077e-2f);
    const __m512 q2 = _mm512_set1_ps(2.272655482e-1f);
    const __m512 q3 = _mm512_set1_ps(2.0f);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512i oneInt = _mm512_set1_epi32(1);
    const __m512i exponentBias = _mm512_set1_epi32(0x7f);

    for (int i = 0; i < n; i += 16) {
        __m512 value = clampExpInput(_mm512_loadu_ps(values + i));
        const __m512 roundedInput = _mm512_fmadd_ps(
            value, reciprocalLn2, half);
        const __mmask16 correctionMask = _mm512_cmp_ps_mask(
            zero, roundedInput, _CMP_NLT_US);
        __m512i exponent = _mm512_cvttps_epi32(roundedInput);
        exponent = _mm512_sub_epi32(exponent,
            _mm512_maskz_mov_epi32(correctionMask, oneInt));
        const __m512 exponentFloat = _mm512_cvtepi32_ps(exponent);
        value = _mm512_fnmadd_ps(exponentFloat, c2, value);
        value = _mm512_fnmadd_ps(exponentFloat, c1, value);

        const __m512 square = _mm512_mul_ps(value, value);
        __m512 denominator = _mm512_fmadd_ps(q0, square, q1);
        __m512 numerator = _mm512_fmadd_ps(p0, square, p1);
        denominator = _mm512_fmadd_ps(denominator, square, q2);
        numerator = _mm512_mul_ps(numerator, square);
        denominator = _mm512_fmadd_ps(denominator, square, q3);
        numerator = _mm512_fmadd_ps(numerator, value, value);
        denominator = _mm512_sub_ps(denominator, numerator);
        const __m512 ratio = _mm512_div_ps(numerator, denominator);
        const __m512 approximation = _mm512_fmadd_ps(ratio, two, one);
        const __m512i scaleBits = _mm512_slli_epi32(
            _mm512_add_epi32(exponent, exponentBias), 23);
        _mm512_storeu_ps(values + i, _mm512_mul_ps(
            approximation, _mm512_castsi512_ps(scaleBits)));
    }
}

void weightedAverageAVX512(const float* weights, const int n, float* mstd)
{
    const float* const outputs = weights + n;
    const __m512 absoluteMask = _mm512_castsi512_ps(
        _mm512_set1_epi32(0x7fffffff));
    const __m512 one = _mm512_set1_ps(1.0f);
    __m512 weightSum = _mm512_setzero_ps();
    __m512 valueSum = _mm512_setzero_ps();

    for (int i = 0; i < n; i += 16) {
        const __m512 weight = _mm512_loadu_ps(weights + i);
        const __m512 output = _mm512_loadu_ps(outputs + i);
        const __m512 denominator = _mm512_add_ps(
            _mm512_and_ps(output, absoluteMask), one);
        const __m512 elliott = _mm512_div_ps(output, denominator);
        weightSum = _mm512_add_ps(weightSum, weight);
        valueSum = _mm512_fmadd_ps(weight, elliott, valueSum);
    }

    const float weightTotal = horizontalSum16(weightSum);
    const float valueTotal = horizontalSum16(valueSum);
    float normalized = 0.0f;
    if (weightTotal > 1.0e-10f) {
        normalized = _mm_cvtss_f32(_mm_div_ss(
            _mm_set_ss(5.0f * valueTotal), _mm_set_ss(weightTotal)));
    }
    const __m128 centered = _mm_fmadd_ss(
        _mm_set_ss(normalized), _mm_set_ss(mstd[1]), _mm_set_ss(mstd[0]));
    mstd[3] += _mm_cvtss_f32(centered);
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

extern "C" NNEDI3_AVX512VNNI_TARGET
void dotProd_m32_m16_i16_AVX512VNNI(const float* data, const float* weights,
    float* vals, const int n, const int len, const float* istd)
{
    dotProdInt16AVX512VNNI(data, weights, vals, n, len, istd);
}

extern "C" NNEDI3_AVX512VNNI_TARGET
void dotProd_m48_m16_i16_AVX512VNNI(const float* data, const float* weights,
    float* vals, const int n, const int len, const float* istd)
{
    dotProdInt16AVX512VNNI(data, weights, vals, n, len, istd);
}

extern "C" void e0_m16_AVX512(float* values, const int n)
{
    expApprox0AVX512(values, n);
}

extern "C" void e1_m16_AVX512(float* values, const int n)
{
    expApprox1AVX512(values, n);
}

extern "C" void e2_m16_AVX512(float* values, const int n)
{
    expApprox2AVX512(values, n);
}

extern "C" void weightedAvgElliottMul5_m16_AVX512(
    const float* weights, const int n, float* mstd)
{
    weightedAverageAVX512(weights, n, mstd);
}

#undef NNEDI3_AVX512VNNI_TARGET

#include "nnedi3_intrinsic_AVXVNNI.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
#define NNEDI3_AVXVNNI_TARGET __attribute__((target("avx2,fma,avxvnni")))
#else
#define NNEDI3_AVXVNNI_TARGET
#endif

namespace {

NNEDI3_AVXVNNI_TARGET
std::int32_t horizontalSum8xInt32(const __m256i value)
{
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
    return _mm_cvtsi128_si32(sum);
}

NNEDI3_AVXVNNI_TARGET
__m128i horizontalSum4x8Int32(const __m256i sum0, const __m256i sum1,
    const __m256i sum2, const __m256i sum3)
{
    return _mm_setr_epi32(horizontalSum8xInt32(sum0),
        horizontalSum8xInt32(sum1), horizontalSum8xInt32(sum2),
        horizontalSum8xInt32(sum3));
}

NNEDI3_AVXVNNI_TARGET
__m128 elliott(const __m128 value)
{
    const __m128 absoluteMask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    return _mm_div_ps(value,
        _mm_add_ps(_mm_and_ps(value, absoluteMask), _mm_set1_ps(1.0f)));
}

NNEDI3_AVXVNNI_TARGET
__m128 accumulateFour(const __m128 input, const float* weights,
    const __m128 bias)
{
    __m128 result = bias;
    result = _mm_fmadd_ps(_mm_shuffle_ps(input, input, 0x00),
        _mm_loadu_ps(weights), result);
    result = _mm_fmadd_ps(_mm_shuffle_ps(input, input, 0x55),
        _mm_loadu_ps(weights + 4), result);
    result = _mm_fmadd_ps(_mm_shuffle_ps(input, input, 0xaa),
        _mm_loadu_ps(weights + 8), result);
    return _mm_fmadd_ps(_mm_shuffle_ps(input, input, 0xff),
        _mm_loadu_ps(weights + 12), result);
}

NNEDI3_AVXVNNI_TARGET
std::uint8_t evaluateOldNetworkTail(const __m128 firstLayer,
    const float* secondWeights, const float* secondBias,
    const float* outputWeights, const float* outputBias)
{
    const __m128 firstActivated = elliott(firstLayer);
    const __m128 firstFeatures = _mm_blend_ps(firstActivated, firstLayer, 0x01);
    const __m128 secondLayer = accumulateFour(
        firstFeatures, secondWeights, _mm_loadu_ps(secondBias));
    const __m128 secondActivated = elliott(secondLayer);

    __m128 output = _mm_loadu_ps(outputBias);
    output = _mm_fmadd_ps(_mm_shuffle_ps(firstFeatures, firstFeatures, 0x00),
        _mm_loadu_ps(outputWeights), output);
    output = _mm_fmadd_ps(_mm_shuffle_ps(firstFeatures, firstFeatures, 0x55),
        _mm_loadu_ps(outputWeights + 4), output);
    output = _mm_fmadd_ps(_mm_shuffle_ps(firstFeatures, firstFeatures, 0xaa),
        _mm_loadu_ps(outputWeights + 8), output);
    output = _mm_fmadd_ps(_mm_shuffle_ps(firstFeatures, firstFeatures, 0xff),
        _mm_loadu_ps(outputWeights + 12), output);
    output = _mm_fmadd_ps(_mm_shuffle_ps(secondActivated, secondActivated, 0x00),
        _mm_loadu_ps(outputWeights + 16), output);
    output = _mm_fmadd_ps(_mm_shuffle_ps(secondActivated, secondActivated, 0x55),
        _mm_loadu_ps(outputWeights + 20), output);
    output = _mm_fmadd_ps(_mm_shuffle_ps(secondActivated, secondActivated, 0xaa),
        _mm_loadu_ps(outputWeights + 24), output);
    output = _mm_fmadd_ps(_mm_shuffle_ps(secondActivated, secondActivated, 0xff),
        _mm_loadu_ps(outputWeights + 28), output);

    // SIMD用の0,2,1,3 lane配置で、後半組の最大値が大きい場合だけ0にする。
    const __m128 highPair = _mm_movehl_ps(output, output);
    const __m128 pairMax = _mm_max_ps(output, highPair);
    const __m128 second = _mm_shuffle_ps(pairMax, pairMax, 0x55);
    return _mm_comigt_ss(second, pairMax) ? 0 : 1;
}

NNEDI3_AVXVNNI_TARGET
__m128i dotProductFour(const std::int16_t* input,
    const std::int16_t* weights, const int length)
{
    __m256i sums0 = _mm256_setzero_si256();
    __m256i sums1 = _mm256_setzero_si256();
    __m256i sums2 = _mm256_setzero_si256();
    __m256i sums3 = _mm256_setzero_si256();
    for (int sample = 0; sample < length; sample += 16) {
        const __m256i values = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(input + sample));
        const std::int16_t* const tile = weights
            + static_cast<std::size_t>(sample) * 4;
        sums0 = _mm256_dpwssd_avx_epi32(sums0, values,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile)));
        sums1 = _mm256_dpwssd_avx_epi32(sums1, values,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile + 16)));
        sums2 = _mm256_dpwssd_avx_epi32(sums2, values,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile + 32)));
        sums3 = _mm256_dpwssd_avx_epi32(sums3, values,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile + 48)));
    }
    return horizontalSum4x8Int32(sums0, sums1, sums2, sums3);
}

NNEDI3_AVXVNNI_TARGET
void dotProdInt16AVXVNNI(const float* dataRaw, const float* weightsRaw,
    float* vals, const int n, const int len, const float* istd)
{
    const auto* const data = reinterpret_cast<const std::int16_t*>(dataRaw);
    const auto* const weights = reinterpret_cast<const std::int16_t*>(weightsRaw);
    const float* const scaleBias = reinterpret_cast<const float*>(
        weights + static_cast<std::size_t>(n) * len);
    const __m128 inverseStdDev = _mm_set1_ps(*istd);

    for (int neuron = 0; neuron < n; neuron += 8) {
        const std::int16_t* const group0 = weights
            + static_cast<std::size_t>(neuron) * len;
        const std::int16_t* const group1 = group0
            + static_cast<std::size_t>(4) * len;
        __m256i sums0 = _mm256_setzero_si256();
        __m256i sums1 = _mm256_setzero_si256();
        __m256i sums2 = _mm256_setzero_si256();
        __m256i sums3 = _mm256_setzero_si256();
        __m256i sums4 = _mm256_setzero_si256();
        __m256i sums5 = _mm256_setzero_si256();
        __m256i sums6 = _mm256_setzero_si256();
        __m256i sums7 = _mm256_setzero_si256();
        for (int sample = 0; sample < len; sample += 16) {
            const __m256i values = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(data + sample));
            const std::int16_t* const tile0 = group0
                + static_cast<std::size_t>(sample) * 4;
            const std::int16_t* const tile1 = group1
                + static_cast<std::size_t>(sample) * 4;
            sums0 = _mm256_dpwssd_avx_epi32(sums0, values,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile0)));
            sums1 = _mm256_dpwssd_avx_epi32(sums1, values,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile0 + 16)));
            sums2 = _mm256_dpwssd_avx_epi32(sums2, values,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile0 + 32)));
            sums3 = _mm256_dpwssd_avx_epi32(sums3, values,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile0 + 48)));
            sums4 = _mm256_dpwssd_avx_epi32(sums4, values,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile1)));
            sums5 = _mm256_dpwssd_avx_epi32(sums5, values,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile1 + 16)));
            sums6 = _mm256_dpwssd_avx_epi32(sums6, values,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile1 + 32)));
            sums7 = _mm256_dpwssd_avx_epi32(sums7, values,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile1 + 48)));
        }
        const __m128 converted0 = _mm_cvtepi32_ps(
            horizontalSum4x8Int32(sums0, sums1, sums2, sums3));
        const __m128 converted1 = _mm_cvtepi32_ps(
            horizontalSum4x8Int32(sums4, sums5, sums6, sums7));
        const float* const groupScaleBias = scaleBias
            + static_cast<std::size_t>(neuron / 4) * 8;
        const __m128 scaled0 = _mm_mul_ps(
            converted0, _mm_loadu_ps(groupScaleBias));
        const __m128 scaled1 = _mm_mul_ps(
            converted1, _mm_loadu_ps(groupScaleBias + 8));
        _mm_storeu_ps(vals + neuron, _mm_fmadd_ps(
            scaled0, inverseStdDev, _mm_loadu_ps(groupScaleBias + 4)));
        _mm_storeu_ps(vals + neuron + 4, _mm_fmadd_ps(
            scaled1, inverseStdDev, _mm_loadu_ps(groupScaleBias + 12)));
    }
    _mm256_zeroupper();
}

} // 無名名前空間

extern "C" NNEDI3_AVXVNNI_TARGET
void computeNetwork0_i16_AVXVNNI(const float* inputRaw,
    const float* weightsRaw, std::uint8_t* result)
{
    const auto* const input = reinterpret_cast<const std::int16_t*>(inputRaw);
    const auto* const weights = reinterpret_cast<const std::int16_t*>(weightsRaw);
    const float* const floatWeights = reinterpret_cast<const float*>(weights + 192);
    const __m128 firstLayer = _mm_fmadd_ps(
        _mm_cvtepi32_ps(dotProductFour(input, weights, 48)),
        _mm_loadu_ps(floatWeights), _mm_loadu_ps(floatWeights + 4));
    *result = evaluateOldNetworkTail(firstLayer, floatWeights + 8,
        floatWeights + 24, floatWeights + 28, floatWeights + 60);
    _mm256_zeroupper();
}

extern "C" NNEDI3_AVXVNNI_TARGET
void computeNetwork0new_AVXVNNI(const float* inputRaw,
    const float* weightsRaw, std::uint8_t* result)
{
    const auto* const input = reinterpret_cast<const std::int16_t*>(inputRaw);
    const auto* const weights = reinterpret_cast<const std::int16_t*>(weightsRaw);
    const float* const floatWeights = reinterpret_cast<const float*>(weights + 256);
    const __m128 firstLayer = elliott(_mm_fmadd_ps(
        _mm_cvtepi32_ps(dotProductFour(input, weights, 64)),
        _mm_loadu_ps(floatWeights), _mm_loadu_ps(floatWeights + 4)));
    const __m128 output = accumulateFour(
        firstLayer, floatWeights + 8, _mm_loadu_ps(floatWeights + 24));
    const unsigned int mask = static_cast<unsigned int>(_mm_movemask_ps(
        _mm_cmp_ps(output, _mm_setzero_ps(), _CMP_NLT_US)));
    const std::uint32_t bytes = ((mask & 0x01u) << 0)
        | ((mask & 0x02u) << 7) | ((mask & 0x04u) << 14)
        | ((mask & 0x08u) << 21);
    std::memcpy(result, &bytes, sizeof(bytes));
    _mm256_zeroupper();
}

extern "C" NNEDI3_AVXVNNI_TARGET
void dotProd_m32_m16_i16_AVXVNNI(const float* data, const float* weights,
    float* vals, const int n, const int len, const float* istd)
{
    dotProdInt16AVXVNNI(data, weights, vals, n, len, istd);
}

extern "C" NNEDI3_AVXVNNI_TARGET
void dotProd_m48_m16_i16_AVXVNNI(const float* data, const float* weights,
    float* vals, const int n, const int len, const float* istd)
{
    dotProdInt16AVXVNNI(data, weights, vals, n, len, istd);
}

#undef NNEDI3_AVXVNNI_TARGET

#include "nnedi3_intrinsic_AVX512_prescreener.h"
#include "nnedi3_intrinsic_AVX512.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
#define NNEDI3_AVX512VNNI_TARGET __attribute__((target("avx512vnni")))
#else
#define NNEDI3_AVX512VNNI_TARGET
#endif

namespace {

constexpr __mmask16 mask12 = 0x0fff;

float horizontalSum16(const __m512 value)
{
    const __m256 sum8 = _mm256_add_ps(
        _mm512_castps512_ps256(value), _mm512_extractf32x8_ps(value, 1));
    __m128 low = _mm256_castps256_ps128(sum8);
    __m128 high = _mm256_extractf128_ps(sum8, 1);
    low = _mm_hadd_ps(low, low);
    high = _mm_hadd_ps(high, high);
    low = _mm_hadd_ps(low, low);
    high = _mm_hadd_ps(high, high);
    return _mm_cvtss_f32(_mm_add_ss(low, high));
}

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

__m128 elliott(const __m128 value)
{
    const __m128 absoluteMask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 denominator = _mm_add_ps(
        _mm_and_ps(value, absoluteMask), _mm_set1_ps(1.0f));
    return _mm_div_ps(value, denominator);
}

__m128 accumulateFour(const __m128 input, const float* weights, const __m128 bias)
{
    __m128 result = bias;
    result = _mm_fmadd_ps(_mm_shuffle_ps(input, input, 0x00),
        _mm_loadu_ps(weights + 0), result);
    result = _mm_fmadd_ps(_mm_shuffle_ps(input, input, 0x55),
        _mm_loadu_ps(weights + 4), result);
    result = _mm_fmadd_ps(_mm_shuffle_ps(input, input, 0xaa),
        _mm_loadu_ps(weights + 8), result);
    result = _mm_fmadd_ps(_mm_shuffle_ps(input, input, 0xff),
        _mm_loadu_ps(weights + 12), result);
    return result;
}

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
        _mm_loadu_ps(outputWeights + 0), output);
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

__m128 dotProductFloat48(const float* input, const float* weights)
{
    __m512 sums0 = _mm512_setzero_ps();
    __m512 sums1 = _mm512_setzero_ps();
    __m512 sums2 = _mm512_setzero_ps();
    __m512 sums3 = _mm512_setzero_ps();
    for (int block = 0; block < 3; ++block) {
        const __m512 values = _mm512_loadu_ps(input + block * 16);
        const float* const blockWeights = weights + block * 64;
        sums0 = _mm512_fmadd_ps(values, _mm512_loadu_ps(blockWeights + 0), sums0);
        sums1 = _mm512_fmadd_ps(values, _mm512_loadu_ps(blockWeights + 16), sums1);
        sums2 = _mm512_fmadd_ps(values, _mm512_loadu_ps(blockWeights + 32), sums2);
        sums3 = _mm512_fmadd_ps(values, _mm512_loadu_ps(blockWeights + 48), sums3);
    }
    return _mm_setr_ps(horizontalSum16(sums0), horizontalSum16(sums1),
        horizontalSum16(sums2), horizontalSum16(sums3));
}

__m128 dotProductInt16(const std::int16_t* input, const std::int16_t* weights,
    const int length)
{
    __m512i sums0 = _mm512_setzero_si512();
    __m512i sums1 = _mm512_setzero_si512();
    __m512i sums2 = _mm512_setzero_si512();
    __m512i sums3 = _mm512_setzero_si512();
    const int fullLength = length & ~31;
    for (int block = 0; block < fullLength; block += 32) {
        const __m512i values = _mm512_loadu_si512(input + block);
        const std::int16_t* const blockWeights = weights
            + static_cast<std::size_t>(block / 32) * 128;
        sums0 = _mm512_add_epi32(sums0,
            _mm512_madd_epi16(values, _mm512_loadu_si512(blockWeights + 0)));
        sums1 = _mm512_add_epi32(sums1,
            _mm512_madd_epi16(values, _mm512_loadu_si512(blockWeights + 32)));
        sums2 = _mm512_add_epi32(sums2,
            _mm512_madd_epi16(values, _mm512_loadu_si512(blockWeights + 64)));
        sums3 = _mm512_add_epi32(sums3,
            _mm512_madd_epi16(values, _mm512_loadu_si512(blockWeights + 96)));
    }
    if (fullLength != length) {
        constexpr __mmask32 tailMask = 0x0000ffff;
        const __m512i values = _mm512_maskz_loadu_epi16(tailMask, input + fullLength);
        const std::int16_t* const tailWeights = weights + 128;
        sums0 = _mm512_add_epi32(sums0, _mm512_madd_epi16(values,
            _mm512_maskz_loadu_epi16(tailMask, tailWeights + 0)));
        sums1 = _mm512_add_epi32(sums1, _mm512_madd_epi16(values,
            _mm512_maskz_loadu_epi16(tailMask, tailWeights + 16)));
        sums2 = _mm512_add_epi32(sums2, _mm512_madd_epi16(values,
            _mm512_maskz_loadu_epi16(tailMask, tailWeights + 32)));
        sums3 = _mm512_add_epi32(sums3, _mm512_madd_epi16(values,
            _mm512_maskz_loadu_epi16(tailMask, tailWeights + 48)));
    }
    return _mm_cvtepi32_ps(_mm_setr_epi32(horizontalSum16xInt32(sums0),
        horizontalSum16xInt32(sums1), horizontalSum16xInt32(sums2),
        horizontalSum16xInt32(sums3)));
}

NNEDI3_AVX512VNNI_TARGET
__m128 dotProductInt16VNNI64(const std::int16_t* input,
    const std::int16_t* weights)
{
    const __m512i values0 = _mm512_loadu_si512(input);
    const __m512i values1 = _mm512_loadu_si512(input + 32);
    __m512i sums0 = _mm512_setzero_si512();
    __m512i sums1 = _mm512_setzero_si512();
    __m512i sums2 = _mm512_setzero_si512();
    __m512i sums3 = _mm512_setzero_si512();
    sums0 = _mm512_dpwssd_epi32(sums0, values0, _mm512_loadu_si512(weights));
    sums1 = _mm512_dpwssd_epi32(sums1, values0, _mm512_loadu_si512(weights + 32));
    sums2 = _mm512_dpwssd_epi32(sums2, values0, _mm512_loadu_si512(weights + 64));
    sums3 = _mm512_dpwssd_epi32(sums3, values0, _mm512_loadu_si512(weights + 96));
    sums0 = _mm512_dpwssd_epi32(sums0, values1, _mm512_loadu_si512(weights + 128));
    sums1 = _mm512_dpwssd_epi32(sums1, values1, _mm512_loadu_si512(weights + 160));
    sums2 = _mm512_dpwssd_epi32(sums2, values1, _mm512_loadu_si512(weights + 192));
    sums3 = _mm512_dpwssd_epi32(sums3, values1, _mm512_loadu_si512(weights + 224));
    return _mm_cvtepi32_ps(horizontalSum4x16Int32(
        sums0, sums1, sums2, sums3));
}

NNEDI3_AVX512VNNI_TARGET
__m128 dotProductInt16VNNI48(const std::int16_t* input,
    const std::int16_t* weights)
{
    constexpr __mmask32 tailMask = 0x0000ffff;
    const __m512i values0 = _mm512_loadu_si512(input);
    const __m512i values1 = _mm512_maskz_loadu_epi16(tailMask, input + 32);
    __m512i sums0 = _mm512_setzero_si512();
    __m512i sums1 = _mm512_setzero_si512();
    __m512i sums2 = _mm512_setzero_si512();
    __m512i sums3 = _mm512_setzero_si512();
    sums0 = _mm512_dpwssd_epi32(sums0, values0, _mm512_loadu_si512(weights));
    sums1 = _mm512_dpwssd_epi32(sums1, values0, _mm512_loadu_si512(weights + 32));
    sums2 = _mm512_dpwssd_epi32(sums2, values0, _mm512_loadu_si512(weights + 64));
    sums3 = _mm512_dpwssd_epi32(sums3, values0, _mm512_loadu_si512(weights + 96));
    sums0 = _mm512_dpwssd_epi32(sums0, values1,
        _mm512_maskz_loadu_epi16(tailMask, weights + 128));
    sums1 = _mm512_dpwssd_epi32(sums1, values1,
        _mm512_maskz_loadu_epi16(tailMask, weights + 144));
    sums2 = _mm512_dpwssd_epi32(sums2, values1,
        _mm512_maskz_loadu_epi16(tailMask, weights + 160));
    sums3 = _mm512_dpwssd_epi32(sums3, values1,
        _mm512_maskz_loadu_epi16(tailMask, weights + 176));
    return _mm_cvtepi32_ps(horizontalSum4x16Int32(
        sums0, sums1, sums2, sums3));
}

NNEDI3_AVX512VNNI_TARGET
void computeNetwork0i16VNNI(const float* inputRaw, const float* weightsRaw,
    std::uint8_t* result)
{
    const auto* input = reinterpret_cast<const std::int16_t*>(inputRaw);
    const auto* weights = reinterpret_cast<const std::int16_t*>(weightsRaw);
    const float* const floatWeights = reinterpret_cast<const float*>(weights + 192);
    const __m128 converted = dotProductInt16VNNI48(input, weights);
    const __m128 firstLayer = _mm_fmadd_ps(converted,
        _mm_loadu_ps(floatWeights), _mm_loadu_ps(floatWeights + 4));
    *result = evaluateOldNetworkTail(firstLayer, floatWeights + 8,
        floatWeights + 24, floatWeights + 28, floatWeights + 60);
}

NNEDI3_AVX512VNNI_TARGET
void computeNetwork0newVNNI(const float* inputRaw, const float* weightsRaw,
    std::uint8_t* result)
{
    const auto* input = reinterpret_cast<const std::int16_t*>(inputRaw);
    const auto* weights = reinterpret_cast<const std::int16_t*>(weightsRaw);
    const float* const floatWeights = reinterpret_cast<const float*>(weights + 256);
    const __m128 converted = dotProductInt16VNNI64(input, weights);
    const __m128 firstLayer = elliott(_mm_fmadd_ps(converted,
        _mm_loadu_ps(floatWeights), _mm_loadu_ps(floatWeights + 4)));
    const __m128 output = accumulateFour(
        firstLayer, floatWeights + 8, _mm_loadu_ps(floatWeights + 24));
    const __mmask8 nonNegative = _mm_cmp_ps_mask(
        output, _mm_setzero_ps(), _CMP_NLT_US);
    const std::uint32_t bytes = ((nonNegative & 0x01u) << 0)
        | ((nonNegative & 0x02u) << 7)
        | ((nonNegative & 0x04u) << 14)
        | ((nonNegative & 0x08u) << 21);
    std::memcpy(result, &bytes, sizeof(bytes));
}

void convertBytesToInt16(const std::uint8_t* source, const int pitch,
    std::int16_t* destination, const int width)
{
    const __mmask16 storeMask = static_cast<__mmask16>((1u << width) - 1u);
    for (int row = 0; row < 4; ++row) {
        const __m128i bytes = _mm_maskz_loadu_epi8(storeMask, source);
        _mm256_mask_storeu_epi16(destination, storeMask, _mm256_cvtepu8_epi16(bytes));
        source += pitch * 2;
        destination += width;
    }
}

} // 無名名前空間

extern "C" void uc2s48_AVX512(
    const std::uint8_t* source, const int pitch, float* destination)
{
    convertBytesToInt16(source, pitch,
        reinterpret_cast<std::int16_t*>(destination), 12);
}

extern "C" void uc2s64_AVX512(
    const std::uint8_t* source, const int pitch, float* destination)
{
    auto* output = reinterpret_cast<std::int16_t*>(destination);
    const std::ptrdiff_t rowStride = static_cast<std::ptrdiff_t>(pitch) * 2;
    const __m256i rows01 = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(source))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + rowStride)), 1);
    const __m256i rows23 = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(source + rowStride * 2))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + rowStride * 3)), 1);
    _mm512_storeu_si512(output, _mm512_cvtepu8_epi16(rows01));
    _mm512_storeu_si512(output + 32, _mm512_cvtepu8_epi16(rows23));
}

extern "C" void uc2s48_AVX512_16(
    const std::uint8_t* source, const int pitch, float* destinationRaw)
{
    auto* destination = reinterpret_cast<std::uint16_t*>(destinationRaw);
    for (int row = 0; row < 4; ++row) {
        const __m256i values = _mm256_maskz_loadu_epi16(
            mask12, reinterpret_cast<const std::uint16_t*>(source));
        _mm256_mask_storeu_epi16(destination, mask12, values);
        source += pitch * 2;
        destination += 12;
    }
}

extern "C" void uc2s64_AVX512_16(
    const std::uint8_t* source, const int pitch, float* destinationRaw)
{
    auto* destination = reinterpret_cast<std::uint16_t*>(destinationRaw);
    for (int row = 0; row < 4; ++row) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source)));
        source += pitch * 2;
        destination += 16;
    }
}

extern "C" void uc2f48_AVX512(
    const std::uint8_t* source, const int pitch, float* destination)
{
    for (int row = 0; row < 4; ++row) {
        const __m128i bytes = _mm_maskz_loadu_epi8(mask12, source);
        const __m512 values = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
        _mm512_mask_storeu_ps(destination, mask12, values);
        source += pitch * 2;
        destination += 12;
    }
}

extern "C" void uc2f48_AVX512_16(
    const std::uint8_t* source, const int pitch, float* destination)
{
    for (int row = 0; row < 4; ++row) {
        const __m256i words = _mm256_maskz_loadu_epi16(
            mask12, reinterpret_cast<const std::uint16_t*>(source));
        const __m512 values = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(words));
        _mm512_mask_storeu_ps(destination, mask12, values);
        source += pitch * 2;
        destination += 12;
    }
}

extern "C" void uc2f48_AVX512_32(
    const std::uint8_t* source, const int pitch, float* destination)
{
    for (int row = 0; row < 4; ++row) {
        const __m512 values = _mm512_maskz_loadu_ps(
            mask12, reinterpret_cast<const float*>(source));
        _mm512_mask_storeu_ps(destination, mask12, values);
        source += pitch * 2;
        destination += 12;
    }
}

extern "C" void computeNetwork0_AVX512(
    const float* input, const float* weights, std::uint8_t* result)
{
    const __m128 firstLayer = _mm_add_ps(
        dotProductFloat48(input, weights), _mm_loadu_ps(weights + 192));
    *result = evaluateOldNetworkTail(firstLayer, weights + 196, weights + 212,
        weights + 216, weights + 248);
}

extern "C" void computeNetwork0_i16_AVX512(
    const float* inputRaw, const float* weightsRaw, std::uint8_t* result)
{
    if (nnedi3_avx512_vnni_supported) {
        computeNetwork0i16VNNI(inputRaw, weightsRaw, result);
        return;
    }
    const auto* input = reinterpret_cast<const std::int16_t*>(inputRaw);
    const auto* weights = reinterpret_cast<const std::int16_t*>(weightsRaw);
    const float* const floatWeights = reinterpret_cast<const float*>(weights + 192);
    const __m128 converted = dotProductInt16(input, weights, 48);
    const __m128 firstLayer = _mm_fmadd_ps(converted,
        _mm_loadu_ps(floatWeights), _mm_loadu_ps(floatWeights + 4));
    *result = evaluateOldNetworkTail(firstLayer, floatWeights + 8,
        floatWeights + 24, floatWeights + 28, floatWeights + 60);
}

extern "C" void computeNetwork0new_AVX512(
    const float* inputRaw, const float* weightsRaw, std::uint8_t* result)
{
    if (nnedi3_avx512_vnni_supported) {
        computeNetwork0newVNNI(inputRaw, weightsRaw, result);
        return;
    }
    const auto* input = reinterpret_cast<const std::int16_t*>(inputRaw);
    const auto* weights = reinterpret_cast<const std::int16_t*>(weightsRaw);
    const float* const floatWeights = reinterpret_cast<const float*>(weights + 256);
    const __m128 converted = dotProductInt16(input, weights, 64);
    const __m128 firstLayer = elliott(_mm_fmadd_ps(converted,
        _mm_loadu_ps(floatWeights), _mm_loadu_ps(floatWeights + 4)));
    const __m128 output = accumulateFour(
        firstLayer, floatWeights + 8, _mm_loadu_ps(floatWeights + 24));
    const __mmask8 nonNegative = _mm_cmp_ps_mask(
        output, _mm_setzero_ps(), _CMP_NLT_US);
    const std::uint32_t bytes = ((nonNegative & 0x01u) << 0)
        | ((nonNegative & 0x02u) << 7)
        | ((nonNegative & 0x04u) << 14)
        | ((nonNegative & 0x08u) << 21);
    std::memcpy(result, &bytes, sizeof(bytes));
}

#undef NNEDI3_AVX512VNNI_TARGET

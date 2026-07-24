#include "nnedi3_intrinsic_AVX512_process.h"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

__mmask64 tailMask64(const int count)
{
    return count >= 64
        ? static_cast<__mmask64>(~UINT64_C(0))
        : static_cast<__mmask64>((UINT64_C(1) << count) - UINT64_C(1));
}

__mmask32 tailMask32(const int count)
{
    return count >= 32
        ? static_cast<__mmask32>(~UINT32_C(0))
        : static_cast<__mmask32>((UINT32_C(1) << count) - UINT32_C(1));
}

__mmask16 tailMask16(const int count)
{
    return count >= 16
        ? static_cast<__mmask16>(UINT16_MAX)
        : static_cast<__mmask16>((UINT32_C(1) << count) - UINT32_C(1));
}

int paddedWidth(const int width, const int unit)
{
    return width <= 0 ? 0 : ((width + unit - 1) / unit) * unit;
}

__m512i addMaskCount(const __m512i sums, const std::uint8_t* mask,
    const __mmask64 active)
{
    const __m512i values = _mm512_maskz_loadu_epi8(active, mask);
    const __m512i differences = _mm512_maskz_mov_epi8(active,
        _mm512_xor_si512(values, _mm512_set1_epi8(1)));
    return _mm512_add_epi64(sums,
        _mm512_sad_epu8(differences, _mm512_setzero_si512()));
}

int finishMaskCount(const __m512i sums)
{
    alignas(64) std::uint64_t lanes[8];
    _mm512_store_si512(lanes, sums);
    std::uint64_t total = 0;
    for (const std::uint64_t lane : lanes) {
        total += lane;
    }
    return static_cast<int>(std::min<std::uint64_t>(
        total, std::numeric_limits<std::uint16_t>::max()));
}

__m256i interpolate8Half(const __m256i src0, const __m256i src2,
    const __m256i src4, const __m256i src6,
    const __m512i minimum, const __m512i maximum)
{
    const __m512i sum24 = _mm512_add_epi16(
        _mm512_cvtepu8_epi16(src2), _mm512_cvtepu8_epi16(src4));
    const __m512i sum06 = _mm512_add_epi16(
        _mm512_cvtepu8_epi16(src0), _mm512_cvtepu8_epi16(src6));
    __m512i result = _mm512_subs_epu16(
        _mm512_mullo_epi16(sum24, _mm512_set1_epi16(19)),
        _mm512_mullo_epi16(sum06, _mm512_set1_epi16(3)));
    result = _mm512_adds_epu16(result, _mm512_set1_epi16(16));
    result = _mm512_srai_epi16(result, 5);
    result = _mm512_min_epi16(result, maximum);
    result = _mm512_max_epi16(result, minimum);
    return _mm512_cvtusepi16_epi8(result);
}

__m256i interpolate16Half(const __m256i src0, const __m256i src2,
    const __m256i src4, const __m256i src6,
    const __m256i minimum, const __m256i maximum)
{
    const __m512i sum24 = _mm512_add_epi32(
        _mm512_cvtepu16_epi32(src2), _mm512_cvtepu16_epi32(src4));
    const __m512i sum06 = _mm512_add_epi32(
        _mm512_cvtepu16_epi32(src0), _mm512_cvtepu16_epi32(src6));
    __m512i result = _mm512_sub_epi32(
        _mm512_mullo_epi32(sum24, _mm512_set1_epi32(19)),
        _mm512_mullo_epi32(sum06, _mm512_set1_epi32(3)));
    result = _mm512_add_epi32(result, _mm512_set1_epi32(16));
    result = _mm512_srai_epi32(result, 5);
    // narrowing前にsigned範囲をclampし、AVX2のvpackusdwと同じ負値ゼロ化を行う。
    result = _mm512_max_epi32(result, _mm512_setzero_si512());
    result = _mm512_min_epi32(result, _mm512_set1_epi32(UINT16_MAX));
    __m256i packed = _mm512_cvtepi32_epi16(result);
    packed = _mm256_min_epu16(packed, maximum);
    return _mm256_max_epu16(packed, minimum);
}

__m512i loadClamp8(const std::uint16_t* values)
{
    alignas(64) static const std::uint16_t clamp_indices[32]{
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        8, 9, 10, 11, 12, 13, 14, 15,
    };
    const __m512i source = _mm512_broadcast_i64x4(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values)));
    return _mm512_permutexvar_epi16(
        _mm512_load_si512(clamp_indices), source);
}

} // 無名名前空間

extern "C" int processLine0_AVX512(const std::uint8_t* mask, const int width,
    std::uint8_t* dst, const std::uint8_t* src3, const int src_pitch,
    const std::uint16_t* min_max)
{
    const int process_width = paddedWidth(width, 32);
    if (process_width == 0) {
        return 0;
    }

    const std::ptrdiff_t pitch = src_pitch;
    const std::uint8_t* const src0 = src3;
    const std::uint8_t* const src2 = src3 + pitch * 2;
    const std::uint8_t* const src4 = src3 + pitch * 4;
    const std::uint8_t* const src6 = src3 + pitch * 6;
    const __m512i minimum = loadClamp8(min_max);
    const __m512i maximum = loadClamp8(min_max + 32);
    __m512i count_sums = _mm512_setzero_si512();

    for (int x = 0; x < process_width; x += 32) {
        const __mmask32 active = tailMask32(32);
        const __m256i output = interpolate8Half(
            _mm256_maskz_loadu_epi8(active, src0 + x),
            _mm256_maskz_loadu_epi8(active, src2 + x),
            _mm256_maskz_loadu_epi8(active, src4 + x),
            _mm256_maskz_loadu_epi8(active, src6 + x), minimum, maximum);
        _mm256_mask_storeu_epi8(dst + x, active, output);
        count_sums = addMaskCount(count_sums, mask + x, tailMask64(32));
    }
    return finishMaskCount(count_sums);
}

extern "C" int processLine0_AVX512_16(const std::uint8_t* mask, const int width,
    std::uint8_t* dst, const std::uint8_t* src3, const int src_pitch,
    const std::uint16_t* min_max)
{
    const int process_width = paddedWidth(width, 16);
    if (process_width == 0) {
        return 0;
    }

    const std::ptrdiff_t pitch = src_pitch;
    const auto* const src0 = reinterpret_cast<const std::uint16_t*>(src3);
    const auto* const src2 = reinterpret_cast<const std::uint16_t*>(src3 + pitch * 2);
    const auto* const src4 = reinterpret_cast<const std::uint16_t*>(src3 + pitch * 4);
    const auto* const src6 = reinterpret_cast<const std::uint16_t*>(src3 + pitch * 6);
    auto* const output_ptr = reinterpret_cast<std::uint16_t*>(dst);
    const __m256i minimum = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(min_max));
    const __m256i maximum = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(min_max + 32));
    __m512i count_sums = _mm512_setzero_si512();

    for (int x = 0; x < process_width; x += 16) {
        const __mmask16 active = tailMask16(16);
        const __m256i output = interpolate16Half(
            _mm256_maskz_loadu_epi16(active, src0 + x),
            _mm256_maskz_loadu_epi16(active, src2 + x),
            _mm256_maskz_loadu_epi16(active, src4 + x),
            _mm256_maskz_loadu_epi16(active, src6 + x), minimum, maximum);
        _mm256_mask_storeu_epi16(output_ptr + x, active, output);
        count_sums = addMaskCount(count_sums, mask + x, tailMask64(16));
    }
    return finishMaskCount(count_sums);
}

extern "C" int processLine0_AVX512_32(const std::uint8_t* mask, const int width,
    std::uint8_t* dst, const std::uint8_t* src3, const int src_pitch)
{
    const int process_width = paddedWidth(width, 8);
    if (process_width == 0) {
        return 0;
    }

    const std::ptrdiff_t pitch = src_pitch;
    const auto* const src0 = reinterpret_cast<const float*>(src3);
    const auto* const src2 = reinterpret_cast<const float*>(src3 + pitch * 2);
    const auto* const src4 = reinterpret_cast<const float*>(src3 + pitch * 4);
    const auto* const src6 = reinterpret_cast<const float*>(src3 + pitch * 6);
    auto* const output_ptr = reinterpret_cast<float*>(dst);
    const __m512 factor19 = _mm512_set1_ps(0.59375f);
    const __m512 factor3 = _mm512_set1_ps(0.09375f);
    __m512i count_sums = _mm512_setzero_si512();

    for (int x = 0; x < process_width; x += 16) {
        const int length = std::min(16, process_width - x);
        const __mmask16 active = tailMask16(length);
        const __m512 sum24 = _mm512_add_ps(
            _mm512_maskz_loadu_ps(active, src2 + x),
            _mm512_maskz_loadu_ps(active, src4 + x));
        const __m512 sum06 = _mm512_add_ps(
            _mm512_maskz_loadu_ps(active, src0 + x),
            _mm512_maskz_loadu_ps(active, src6 + x));
        const __m512 positive = _mm512_mul_ps(sum24, factor19);
        _mm512_mask_storeu_ps(output_ptr + x, active,
            _mm512_fnmadd_ps(sum06, factor3, positive));
        count_sums = addMaskCount(count_sums, mask + x, tailMask64(length));
    }
    return finishMaskCount(count_sums);
}

namespace {

std::int32_t castScaleValue(const float* values, const float* scale,
    const std::uint32_t minimum, const std::uint32_t maximum)
{
    const __m128 rounded = _mm_fmadd_ss(
        _mm_load_ss(values + 3), _mm_load_ss(scale), _mm_set_ss(0.5f));
    const std::int32_t converted = _mm_cvttss_si32(rounded);
    return std::clamp(converted,
        static_cast<std::int32_t>(minimum), static_cast<std::int32_t>(maximum));
}

} // 無名名前空間

extern "C" void castScale_AVX512(const float* values, const float* scale,
    std::uint8_t* dst, const std::uint32_t minimum, const std::uint32_t maximum)
{
    *dst = static_cast<std::uint8_t>(
        castScaleValue(values, scale, minimum, maximum));
}

extern "C" void castScale_AVX512_16(const float* values, const float* scale,
    std::uint16_t* dst, const std::uint32_t minimum, const std::uint32_t maximum)
{
    *dst = static_cast<std::uint16_t>(
        castScaleValue(values, scale, minimum, maximum));
}

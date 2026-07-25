#include "nnedi3_intrinsic_AVX512_pixel_convert.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

__m128i extractLane128(const __m512i value, const int lane)
{
    switch (lane) {
    case 0: return _mm512_castsi512_si128(value);
    case 1: return _mm512_extracti32x4_epi32(value, 1);
    case 2: return _mm512_extracti32x4_epi32(value, 2);
    default: return _mm512_extracti32x4_epi32(value, 3);
    }
}

__m512i insertLane128(__m512i value, const __m128i lane_value, const int lane)
{
    switch (lane) {
    case 0: return _mm512_castsi128_si512(lane_value);
    case 1: return _mm512_inserti32x4(value, lane_value, 1);
    case 2: return _mm512_inserti32x4(value, lane_value, 2);
    default: return _mm512_inserti32x4(value, lane_value, 3);
    }
}

void store12(std::uint8_t* dst, const __m128i value)
{
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), value);
    const std::uint32_t upper = static_cast<std::uint32_t>(
        _mm_extract_epi32(value, 2));
    std::memcpy(dst + 8, &upper, sizeof(upper));
}

} // 無名名前空間

extern "C" void convYUY2to422_AVX512(const std::uint8_t* src,
    std::uint8_t* plane_y, std::uint8_t* plane_u, std::uint8_t* plane_v,
    const int src_pitch, const int y_pitch, const int uv_pitch,
    const int width, const int height)
{
    if (width <= 0 || height <= 0) {
        return;
    }
    alignas(64) static const std::uint8_t y_indices[64]{
        0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    };
    alignas(64) static const std::uint8_t u_indices[64]{
        1, 5, 9, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        1, 5, 9, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        1, 5, 9, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        1, 5, 9, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    };
    alignas(64) static const std::uint8_t v_indices[64]{
        3, 7, 11, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        3, 7, 11, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        3, 7, 11, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        3, 7, 11, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    };
    const __m512i y_shuffle = _mm512_load_si512(y_indices);
    const __m512i u_shuffle = _mm512_load_si512(u_indices);
    const __m512i v_shuffle = _mm512_load_si512(v_indices);

    for (int row = 0; row < height; ++row) {
        int x = 0;
        for (; x + 32 <= width; x += 32) {
            const __m512i packed = _mm512_loadu_si512(src + x * 2);
            const __m512i y_values = _mm512_shuffle_epi8(packed, y_shuffle);
            const __m512i u_values = _mm512_shuffle_epi8(packed, u_shuffle);
            const __m512i v_values = _mm512_shuffle_epi8(packed, v_shuffle);
            for (int lane = 0; lane < 4; ++lane) {
                _mm_storel_epi64(reinterpret_cast<__m128i*>(plane_y + x + lane * 8),
                    extractLane128(y_values, lane));
                const std::uint32_t u = static_cast<std::uint32_t>(
                    _mm_cvtsi128_si32(extractLane128(u_values, lane)));
                const std::uint32_t v = static_cast<std::uint32_t>(
                    _mm_cvtsi128_si32(extractLane128(v_values, lane)));
                std::memcpy(plane_u + x / 2 + lane * 4, &u, sizeof(u));
                std::memcpy(plane_v + x / 2 + lane * 4, &v, sizeof(v));
            }
        }
        for (; x + 1 < width; x += 2) {
            plane_y[x] = src[x * 2];
            plane_u[x / 2] = src[x * 2 + 1];
            plane_y[x + 1] = src[x * 2 + 2];
            plane_v[x / 2] = src[x * 2 + 3];
        }
        src += src_pitch;
        plane_y += y_pitch;
        plane_u += uv_pitch;
        plane_v += uv_pitch;
    }
}

extern "C" void conv422toYUY2_AVX512(const std::uint8_t* plane_y,
    const std::uint8_t* plane_u, const std::uint8_t* plane_v,
    std::uint8_t* dst, const int y_pitch, const int uv_pitch,
    const int dst_pitch, const int width, const int height)
{
    if (width <= 0 || height <= 0) {
        return;
    }
    for (int row = 0; row < height; ++row) {
        int x = 0;
        for (; x + 32 <= width; x += 32) {
            __m512i packed = _mm512_setzero_si512();
            for (int lane = 0; lane < 4; ++lane) {
                const __m128i y = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(plane_y + x + lane * 8));
                std::uint32_t u_value;
                std::uint32_t v_value;
                std::memcpy(&u_value, plane_u + x / 2 + lane * 4, sizeof(u_value));
                std::memcpy(&v_value, plane_v + x / 2 + lane * 4, sizeof(v_value));
                const __m128i u = _mm_cvtsi32_si128(static_cast<int>(u_value));
                const __m128i v = _mm_cvtsi32_si128(static_cast<int>(v_value));
                const __m128i lane_values = _mm_unpacklo_epi8(y, _mm_unpacklo_epi8(u, v));
                packed = insertLane128(packed, lane_values, lane);
            }
            _mm512_storeu_si512(dst + x * 2, packed);
        }
        for (; x + 1 < width; x += 2) {
            dst[x * 2] = plane_y[x];
            dst[x * 2 + 1] = plane_u[x / 2];
            dst[x * 2 + 2] = plane_y[x + 1];
            dst[x * 2 + 3] = plane_v[x / 2];
        }
        plane_y += y_pitch;
        plane_u += uv_pitch;
        plane_v += uv_pitch;
        dst += dst_pitch;
    }
}

extern "C" void convRGB24to444_AVX512(const std::uint8_t* src,
    std::uint8_t* plane_0, std::uint8_t* plane_1, std::uint8_t* plane_2,
    const int src_pitch, const int plane_0_pitch, const int plane_12_pitch,
    const int width, const int height)
{
    if (width <= 0 || height <= 0) {
        return;
    }
    alignas(64) static const std::int32_t gather_indices[16]{
        0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
    };
    const __m512i indices = _mm512_load_si512(gather_indices);
    alignas(64) std::uint8_t tail[64]{};

    for (int row = 0; row < height; ++row) {
        int x = 0;
        for (; x + 16 <= width; x += 16) {
            const std::uint8_t* block = src + x * 3;
            if (x + 16 == width) {
                _mm512_store_si512(tail, _mm512_maskz_loadu_epi8(
                    static_cast<__mmask64>((UINT64_C(1) << 48) - 1), block));
                block = tail;
            }
            const __m512i pixels = _mm512_i32gather_epi32(indices, block, 1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(plane_0 + x),
                _mm512_cvtepi32_epi8(pixels));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(plane_1 + x),
                _mm512_cvtepi32_epi8(_mm512_srli_epi32(pixels, 8)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(plane_2 + x),
                _mm512_cvtepi32_epi8(_mm512_srli_epi32(pixels, 16)));
        }
        for (; x < width; ++x) {
            plane_0[x] = src[x * 3];
            plane_1[x] = src[x * 3 + 1];
            plane_2[x] = src[x * 3 + 2];
        }
        src += src_pitch;
        plane_0 += plane_0_pitch;
        plane_1 += plane_12_pitch;
        plane_2 += plane_12_pitch;
    }
}

extern "C" void conv444toRGB24_AVX512(const std::uint8_t* plane_0,
    const std::uint8_t* plane_1, const std::uint8_t* plane_2,
    std::uint8_t* dst, const int plane_0_pitch, const int plane_12_pitch,
    const int dst_pitch, const int width, const int height)
{
    if (width <= 0 || height <= 0) {
        return;
    }
    alignas(64) static const std::uint8_t compact_indices[64]{
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80,
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80,
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80,
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80,
    };
    const __m512i compact = _mm512_load_si512(compact_indices);
    dst += static_cast<std::ptrdiff_t>(height - 1) * dst_pitch;

    for (int row = 0; row < height; ++row) {
        int x = 0;
        for (; x + 16 <= width; x += 16) {
            const __m512i p0 = _mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(plane_0 + x)));
            const __m512i p1 = _mm512_slli_epi32(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(plane_1 + x))), 8);
            const __m512i p2 = _mm512_slli_epi32(_mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(plane_2 + x))), 16);
            const __m512i packed = _mm512_shuffle_epi8(
                _mm512_or_si512(_mm512_or_si512(p0, p1), p2), compact);
            for (int lane = 0; lane < 4; ++lane) {
                store12(dst + x * 3 + lane * 12, extractLane128(packed, lane));
            }
        }
        for (; x < width; ++x) {
            dst[x * 3] = plane_0[x];
            dst[x * 3 + 1] = plane_1[x];
            dst[x * 3 + 2] = plane_2[x];
        }
        plane_0 += plane_0_pitch;
        plane_1 += plane_12_pitch;
        plane_2 += plane_12_pitch;
        if (row + 1 < height) {
            dst -= dst_pitch;
        }
    }
}

#pragma once

#include <cstdint>

extern "C" {

void convYUY2to422_AVX512(const std::uint8_t* src,
    std::uint8_t* plane_y, std::uint8_t* plane_u, std::uint8_t* plane_v,
    int src_pitch, int y_pitch, int uv_pitch, int width, int height);

void conv422toYUY2_AVX512(const std::uint8_t* plane_y,
    const std::uint8_t* plane_u, const std::uint8_t* plane_v,
    std::uint8_t* dst, int y_pitch, int uv_pitch, int dst_pitch,
    int width, int height);

void convRGB24to444_AVX512(const std::uint8_t* src,
    std::uint8_t* plane_0, std::uint8_t* plane_1, std::uint8_t* plane_2,
    int src_pitch, int plane_0_pitch, int plane_12_pitch,
    int width, int height);

void conv444toRGB24_AVX512(const std::uint8_t* plane_0,
    const std::uint8_t* plane_1, const std::uint8_t* plane_2,
    std::uint8_t* dst, int plane_0_pitch, int plane_12_pitch,
    int dst_pitch, int width, int height);

}

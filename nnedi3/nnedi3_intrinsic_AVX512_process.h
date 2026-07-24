#pragma once

#include <cstdint>

extern "C" {

int processLine0_AVX512(const std::uint8_t* mask, int width,
    std::uint8_t* dst, const std::uint8_t* src3, int src_pitch,
    const std::uint16_t* min_max);

int processLine0_AVX512_16(const std::uint8_t* mask, int width,
    std::uint8_t* dst, const std::uint8_t* src3, int src_pitch,
    const std::uint16_t* min_max);

int processLine0_AVX512_32(const std::uint8_t* mask, int width,
    std::uint8_t* dst, const std::uint8_t* src3, int src_pitch);

void castScale_AVX512(const float* values, const float* scale,
    std::uint8_t* dst, std::uint32_t minimum, std::uint32_t maximum);

void castScale_AVX512_16(const float* values, const float* scale,
    std::uint16_t* dst, std::uint32_t minimum, std::uint32_t maximum);

}

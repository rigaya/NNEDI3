#pragma once

#include <cstdint>

extern "C" {

void extract_m8_AVX512(const std::uint8_t* src, int stride, int xdia, int ydia,
    float* mstd, float* input);
void extract_m8_AVX512_16(const std::uint8_t* src, int stride, int xdia, int ydia,
    float* mstd, float* input);
void extract_m8_AVX512_32(const std::uint8_t* src, int stride, int xdia, int ydia,
    float* mstd, float* input);

void extract_m8_i16_AVX512(const std::uint8_t* src, int stride, int xdia, int ydia,
    float* mstd, float* input);
void extract_m8_i16_AVX512_16(const std::uint8_t* src, int stride, int xdia, int ydia,
    float* mstd, float* input);
void extract_m8_i16_AVX512_16_10(const std::uint8_t* src, int stride, int xdia, int ydia,
    float* mstd, float* input);

}

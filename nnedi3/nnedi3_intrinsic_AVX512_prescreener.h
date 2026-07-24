#pragma once

#include <cstdint>

extern "C" {

void uc2s48_AVX512(const std::uint8_t* source, int pitch, float* destination);
void uc2s64_AVX512(const std::uint8_t* source, int pitch, float* destination);
void uc2s48_AVX512_16(const std::uint8_t* source, int pitch, float* destination);
void uc2s64_AVX512_16(const std::uint8_t* source, int pitch, float* destination);
void uc2f48_AVX512(const std::uint8_t* source, int pitch, float* destination);
void uc2f48_AVX512_16(const std::uint8_t* source, int pitch, float* destination);
void uc2f48_AVX512_32(const std::uint8_t* source, int pitch, float* destination);

void computeNetwork0_AVX512(
    const float* input, const float* weights, std::uint8_t* result);
void computeNetwork0_i16_AVX512(
    const float* input, const float* weights, std::uint8_t* result);
void computeNetwork0new_AVX512(
    const float* input, const float* weights, std::uint8_t* result);

}

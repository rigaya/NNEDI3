#pragma once

#include <cstdint>

extern "C" {

void computeNetwork0_i16_AVXVNNI(
    const float* input, const float* weights, std::uint8_t* result);
void computeNetwork0new_AVXVNNI(
    const float* input, const float* weights, std::uint8_t* result);
void dotProd_m32_m16_i16_AVXVNNI(const float* data, const float* weights,
    float* vals, int n, int len, const float* istd);
void dotProd_m48_m16_i16_AVXVNNI(const float* data, const float* weights,
    float* vals, int n, int len, const float* istd);

}

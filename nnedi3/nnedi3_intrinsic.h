#pragma once

#include <cstdint>

extern "C" {

void computeNetwork0_FMA3(const float* input, const float* weights, uint8_t* d);
void computeNetwork0_i16_AVX2(const float* input, const float* weights, uint8_t* d);
void computeNetwork0new_AVX2(const float* input, const float* weights, uint8_t* d);

void uc2f48_AVX2(const uint8_t* src, int pitch, float* dst);
void uc2f48_AVX2_16(const uint8_t* src, int pitch, float* dst);
void uc2s48_AVX2(const uint8_t* src, int pitch, float* dst);
void uc2s64_AVX2(const uint8_t* src, int pitch, float* dst);

void dotProd_m32_m16_FMA3(const float* data, const float* weights, float* vals, int n, int len, const float* istd);
void dotProd_m48_m16_FMA3(const float* data, const float* weights, float* vals, int n, int len, const float* istd);
void dotProd_m32_m16_i16_AVX2(const float* data, const float* weights, float* vals, int n, int len, const float* istd);
void dotProd_m48_m16_i16_AVX2(const float* data, const float* weights, float* vals, int n, int len, const float* istd);

void e0_m16_FMA3(float* values, int n);
void e1_m16_AVX2(float* values, int n);
void e2_m16_AVX2(float* values, int n);

int processLine0_AVX2_ASM(const uint8_t* mask, int width, uint8_t* dst, const uint8_t* src, int src_pitch, const uint16_t* min_max);
int processLine0_AVX2_ASM_16(const uint8_t* mask, int width, uint8_t* dst, const uint8_t* src, int src_pitch, const uint16_t* min_max);
int processLine0_AVX2_ASM_32(const uint8_t* mask, int width, uint8_t* dst, const uint8_t* src, int src_pitch);

void weightedAvgElliottMul5_m16_FMA3(const float* weights, int n, float* mstd);

void extract_m8_FMA3(const uint8_t* src, int stride, int xdia, int ydia, float* mstd, float* input);
void extract_m8_i16_AVX2(const uint8_t* src, int stride, int xdia, int ydia, float* mstd, float* input);
void extract_m8_i16_AVX2_16(const uint8_t* src, int stride, int xdia, int ydia, float* mstd, float* input);
void extract_m8_i16_AVX2_16_2(const uint8_t* src, int stride, int xdia, int ydia, float* input, int32_t* sum, int64_t* sumsq);
void extract_m8_FMA3_16(const uint8_t* src, int stride, int xdia, int ydia, float* mstd, float* input);
void extract_m8_FMA3_32(const uint8_t* src, int stride, int xdia, int ydia, float* mstd, float* input);

}

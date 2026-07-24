#include <immintrin.h>
#include <stdint.h>
#include <cmath> // For fabsf, expf, etc. if needed for some fallback logic, though intrinsics are preferred.
#include <cstring>
#include "nnedi3_intrinsic.h"

// Intrinsics friendly data definitions
namespace {

// From .data section
const float FLT_EPSILON = 1.192092896e-07f;

// align 16
// sign_bits_f_zero_l qword 7FFFFFFF00000000h,7FFFFFFF7FFFFFFFh
// This is loaded as XMMWORD, meaning 128 bits.
// low qword: 0x7FFFFFFF00000000 -> low dword: 0x00000000, high dword: 0x7FFFFFFF
// high qword: 0x7FFFFFFF7FFFFFFF -> low dword: 0x7FFFFFFF, high dword: 0x7FFFFFFF
// So, as epi32: {0x00000000, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF} (little endian for set_epi32)
alignas(16) const __m128 sign_bits_f_zero_l = _mm_castsi128_ps(_mm_set_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x00000000));

// sign_bits_f qword 2 dup(7FFFFFFF7FFFFFFFh)
// This means 4 dwords of 0x7FFFFFFF
alignas(16) const __m128 sign_bits_f = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

// ones_f real4 4 dup(1.0)
alignas(16) const __m128 ones_f = _mm_set1_ps(1.0f);

// flt_epsilon_sse real4 4 dup(FLT_EPSILON)
alignas(16) const __m128 flt_epsilon_sse = _mm_set1_ps(FLT_EPSILON);

// min_weight_sum real4 4 dup(1.0e-10)
alignas(16) const __m128 min_weight_sum = _mm_set1_ps(1.0e-10f);
// five_f real4 4 dup(5.0)
alignas(16) const __m128 five_f = _mm_set1_ps(5.0f);

// sse_half real4 4 dup(0.5)
alignas(16) const __m128 sse_half = _mm_set1_ps(0.5f);

// data segment align(32)
// exp_hi real4 8 dup(80.0)
alignas(32) const __m256 exp_hi = _mm256_set1_ps(80.0f);
// exp_lo real4 8 dup(-80.0)
alignas(32) const __m256 exp_lo = _mm256_set1_ps(-80.0f);

// e0_mult real4 8 dup(12102203.161561486)
alignas(32) const __m256 e0_mult = _mm256_set1_ps(12102203.161561486f);
// e0_bias real4 8 dup(1064866805.0)
alignas(32) const __m256 e0_bias = _mm256_set1_ps(1064866805.0f);

// e1_scale real4 8 dup(1.4426950409)
alignas(32) const __m256 e1_scale = _mm256_set1_ps(1.4426950409f);
// e1_bias real4 8 dup(12582912.0)
alignas(32) const __m256 e1_bias_ps = _mm256_set1_ps(12582912.0f); // For ps operations
alignas(32) const __m256i e1_bias_si = _mm256_set1_epi32(12582912);    // For integer interpretation (3<<22)

// e1_c1 real4 8 dup(0.701277797)
alignas(32) const __m256 e1_c1 = _mm256_set1_ps(0.701277797f);
// e1_c2 real4 8 dup(0.237348593)
alignas(32) const __m256 e1_c2 = _mm256_set1_ps(0.237348593f);
// e1_c0 real4 8 dup(1.00035)
alignas(32) const __m256 e1_c0 = _mm256_set1_ps(1.00035f);

// exp_rln2 real4 8 dup(1.442695041)
alignas(32) const __m256 exp_rln2 = _mm256_set1_ps(1.442695041f);
// am_0p5 real4 8 dup(0.5)
alignas(32) const __m256 am_0p5 = _mm256_set1_ps(0.5f);
// epi32_1 sdword 8 dup(1)
alignas(32) const __m256i epi32_1 = _mm256_set1_epi32(1);
// exp_c2 real4 8 dup(1.428606820e-6)
alignas(32) const __m256 exp_c2 = _mm256_set1_ps(1.428606820e-6f);
// exp_c1 real4 8 dup(6.931457520e-1)
alignas(32) const __m256 exp_c1 = _mm256_set1_ps(6.931457520e-1f);
// exp_q0 real4 8 dup(3.001985051e-6)
alignas(32) const __m256 exp_q0 = _mm256_set1_ps(3.001985051e-6f);
// exp_p0 real4 8 dup(1.261771931e-4)
alignas(32) const __m256 exp_p0 = _mm256_set1_ps(1.261771931e-4f);
// epi32_0x7f sdword 8 dup(7Fh)
alignas(32) const __m256i epi32_0x7f = _mm256_set1_epi32(0x7F);
// exp_q1 real4 8 dup(2.524483403e-3)
alignas(32) const __m256 exp_q1 = _mm256_set1_ps(2.524483403e-3f);
// exp_p1 real4 8 dup(3.029944077e-2)
alignas(32) const __m256 exp_p1 = _mm256_set1_ps(3.029944077e-2f);
// exp_q2 real4 8 dup(2.272655482e-1)
alignas(32) const __m256 exp_q2 = _mm256_set1_ps(2.272655482e-1f);
// am_1 real4 8 dup(1.0)
alignas(32) const __m256 am_1 = _mm256_set1_ps(1.0f);
// exp_q3 real4 8 dup(2.0)
alignas(32) const __m256 exp_q3 = _mm256_set1_ps(2.0f);

// w_19 sword 16 dup(19)
alignas(32) const __m256i w_19 = _mm256_set1_epi16(19);
// w_3 sword 16 dup(3)
alignas(32) const __m256i w_3 = _mm256_set1_epi16(3);
// uw_16 word 16 dup(16)
alignas(32) const __m256i uw_16 = _mm256_set1_epi16(16); // unsigned, but stored in signed type for intrinsics
// ub_1 byte 32 dup(1)
alignas(32) const __m256i ub_1 = _mm256_set1_epi8(1);

// d_19 sdword 8 dup(19)
alignas(32) const __m256i d_19 = _mm256_set1_epi32(19);
// d_3 sdword 8 dup(3)
alignas(32) const __m256i d_3 = _mm256_set1_epi32(3);
// ud_16 dword 16 dup(16) -> This looks like a typo, dword is 32-bit, so 8 elements for YMM. If it is 16 elements of 16-bit, it's uw_16.
// Assuming it means 8 dwords of 16.
alignas(32) const __m256i ud_16 = _mm256_set1_epi32(16); // unsigned, but stored in signed type
// uw_1 word 16 dup(1)
alignas(32) const __m256i uw_1 = _mm256_set1_epi16(1);

// f_19 real4 8 dup(0.59375)  (19/32)
alignas(32) const __m256 f_19 = _mm256_set1_ps(0.59375f);
// f_3 real4 8 dup(0.09375)   (3/32)
alignas(32) const __m256 f_3 = _mm256_set1_ps(0.09375f);

// sign_bits_f_32 qword 4 dup(7FFFFFFF7FFFFFFFh)
// This means 8 dwords of 0x7FFFFFFF for a YMM register
alignas(32) const __m256 sign_bits_f_32 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
// ones_f_32 real4 8 dup(1.0)
alignas(32) const __m256 ones_f_32 = _mm256_set1_ps(1.0f);

alignas(32) static const __m256i w_19_m256i = _mm256_set1_epi16(19);
alignas(32) static const __m256i w_3_m256i = _mm256_set1_epi16(3);
alignas(32) static const __m256i ub_1_m256i = _mm256_set1_epi8(1);
alignas(32) static const __m256i uw_16_m256i = _mm256_set1_epi16(16);

} // anonymous namespace



// computeNetwork0_FMA3 proc input:dword,weights:dword,ptr_d:dword
// input = rcx
// weights = rdx
// ptr_d = r8
extern "C" void computeNetwork0_FMA3(const float *input, const float *weights, uint8_t *ptr_d) {
    // sub rsp,32
    // .allocstack 32
    // vmovdqu XMMWORD ptr[rsp],xmm6
    // .savexmm128 xmm6,0
    // vmovdqu XMMWORD ptr[rsp+16],xmm7
    // .savexmm128 xmm7,16
    // .endprolog
    
    // mov rax,1
    bool result = true;
    
    // vmovaps ymm4,YMMWORD ptr [rcx]
    __m256 ymm4 = _mm256_load_ps(input);
    // vmulps ymm0,ymm4,YMMWORD ptr [rdx]
    __m256 ymm0 = _mm256_mul_ps(ymm4, _mm256_load_ps(weights));
    // vmulps ymm1,ymm4,YMMWORD ptr [rdx+32]
    __m256 ymm1 = _mm256_mul_ps(ymm4, _mm256_load_ps(weights + 8));
    // vmulps ymm2,ymm4,YMMWORD ptr [rdx+64]
    __m256 ymm2 = _mm256_mul_ps(ymm4, _mm256_load_ps(weights + 16));
    // vmulps ymm3,ymm4,YMMWORD ptr [rdx+96]
    __m256 ymm3 = _mm256_mul_ps(ymm4, _mm256_load_ps(weights + 24));
    
    // vmovaps ymm4,YMMWORD ptr [rcx+32]
    ymm4 = _mm256_load_ps(input + 8);
    
    // vfmadd231ps ymm0,ymm4,YMMWORD ptr [rdx+128]
    ymm0 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 32), ymm0);
    // vfmadd231ps ymm1,ymm4,YMMWORD ptr [rdx+160]
    ymm1 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 40), ymm1);
    // vfmadd231ps ymm2,ymm4,YMMWORD ptr [rdx+192]
    ymm2 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 48), ymm2);
    // vfmadd231ps ymm3,ymm4,YMMWORD ptr [rdx+224]
    ymm3 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 56), ymm3);
    
    // vmovaps ymm4,YMMWORD ptr [rcx+64]
    ymm4 = _mm256_load_ps(input + 16);
    
    // vfmadd231ps ymm0,ymm4,YMMWORD ptr [rdx+256]
    ymm0 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 64), ymm0);
    // vfmadd231ps ymm1,ymm4,YMMWORD ptr [rdx+288]
    ymm1 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 72), ymm1);
    // vfmadd231ps ymm2,ymm4,YMMWORD ptr [rdx+320]
    ymm2 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 80), ymm2);
    // vfmadd231ps ymm3,ymm4,YMMWORD ptr [rdx+352]
    ymm3 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 88), ymm3);
    
    // vmovaps ymm4,YMMWORD ptr [rcx+96]
    ymm4 = _mm256_load_ps(input + 24);
    
    // vfmadd231ps ymm0,ymm4,YMMWORD ptr [rdx+384]
    ymm0 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 96), ymm0);
    // vfmadd231ps ymm1,ymm4,YMMWORD ptr [rdx+416]
    ymm1 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 104), ymm1);
    // vfmadd231ps ymm2,ymm4,YMMWORD ptr [rdx+448]
    ymm2 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 112), ymm2);
    // vfmadd231ps ymm3,ymm4,YMMWORD ptr [rdx+480]
    ymm3 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 120), ymm3);
    
    // vmovaps ymm4,YMMWORD ptr [rcx+128]
    ymm4 = _mm256_load_ps(input + 32);
    
    // vfmadd231ps ymm0,ymm4,YMMWORD ptr [rdx+512]
    ymm0 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 128), ymm0);
    // vfmadd231ps ymm1,ymm4,YMMWORD ptr [rdx+544]
    ymm1 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 136), ymm1);
    // vfmadd231ps ymm2,ymm4,YMMWORD ptr [rdx+576]
    ymm2 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 144), ymm2);
    // vfmadd231ps ymm3,ymm4,YMMWORD ptr [rdx+608]
    ymm3 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 152), ymm3);
    
    // vmovaps ymm4,YMMWORD ptr [rcx+160]
    ymm4 = _mm256_load_ps(input + 40);
    
    // vfmadd231ps ymm0,ymm4,YMMWORD ptr [rdx+640]
    ymm0 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 160), ymm0);
    // vfmadd231ps ymm1,ymm4,YMMWORD ptr [rdx+672]
    ymm1 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 168), ymm1);
    // vfmadd231ps ymm2,ymm4,YMMWORD ptr [rdx+704]
    ymm2 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 176), ymm2);
    // vfmadd231ps ymm3,ymm4,YMMWORD ptr [rdx+736]
    ymm3 = _mm256_fmadd_ps(ymm4, _mm256_load_ps(weights + 184), ymm3);
    
    // vhaddps ymm0,ymm0,ymm1
    ymm0 = _mm256_hadd_ps(ymm0, ymm1);
    // vhaddps ymm2,ymm2,ymm3
    ymm2 = _mm256_hadd_ps(ymm2, ymm3);
    // vhaddps ymm0,ymm0,ymm2
    ymm0 = _mm256_hadd_ps(ymm0, ymm2);
    
    // vextractf128 xmm4,ymm0,1
    __m128 xmm4 = _mm256_extractf128_ps(ymm0, 1);
    // vaddps xmm0,xmm0,xmm4
    __m128 xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm4);
    
    // vaddps xmm0,xmm0,XMMWORD ptr [rdx+768]
    xmm0 = _mm_add_ps(xmm0, _mm_load_ps(weights + 192));
    
    // vmovaps xmm1,xmm0
    __m128 xmm1 = xmm0;
    // vandps xmm0,xmm0,XMMWORD ptr sign_bits_f_zero_l
    xmm0 = _mm_and_ps(xmm0, sign_bits_f_zero_l);
    // vaddps xmm0,xmm0,XMMWORD ptr ones_f
    xmm0 = _mm_add_ps(xmm0, ones_f);
    // vrcpps xmm0,xmm0
    xmm0 = _mm_rcp_ps(xmm0);
    // vmulps xmm0,xmm0,xmm1
    xmm0 = _mm_mul_ps(xmm0, xmm1);
    
    // vpshufd xmm1,xmm0,0
    xmm1 = _mm_shuffle_ps(xmm0, xmm0, 0x00);
    // vpshufd xmm2,xmm0,85
    __m128 xmm2 = _mm_shuffle_ps(xmm0, xmm0, 0x55);
    // vpshufd xmm3,xmm0,170
    __m128 xmm3 = _mm_shuffle_ps(xmm0, xmm0, 0xAA);
    // vpshufd xmm4,xmm0,255
    xmm4 = _mm_shuffle_ps(xmm0, xmm0, 0xFF);
    
    // vmulps xmm1,xmm1,XMMWORD ptr [rdx+784]
    xmm1 = _mm_mul_ps(xmm1, _mm_load_ps(weights + 196));
    // vfmadd231ps xmm1,xmm2,XMMWORD ptr [rdx+784+16]
    xmm1 = _mm_fmadd_ps(xmm2, _mm_load_ps(weights + 200), xmm1);
    // vmulps xmm3,xmm3,XMMWORD ptr [rdx+784+32]
    xmm3 = _mm_mul_ps(xmm3, _mm_load_ps(weights + 204));
    // vfmadd231ps xmm3,xmm4,XMMWORD ptr [rdx+784+48]
    xmm3 = _mm_fmadd_ps(xmm4, _mm_load_ps(weights + 208), xmm3);
    // vaddps xmm1,xmm1,xmm3
    xmm1 = _mm_add_ps(xmm1, xmm3);
    // vaddps xmm1,xmm1,XMMWORD ptr [rdx+784+64]
    xmm1 = _mm_add_ps(xmm1, _mm_load_ps(weights + 212));
    
    // vmovaps xmm7,xmm1
    __m128 xmm7 = xmm1;
    // vandps xmm1,xmm1,XMMWORD ptr sign_bits_f
    xmm1 = _mm_and_ps(xmm1, sign_bits_f);
    // vmovaps xmm3,xmm0
    xmm3 = xmm0;
    // vaddps xmm1,xmm1,XMMWORD ptr ones_f
    xmm1 = _mm_add_ps(xmm1, ones_f);
    // vrcpps xmm1,xmm1
    xmm1 = _mm_rcp_ps(xmm1);
    // vmulps xmm7,xmm7,xmm1
    xmm7 = _mm_mul_ps(xmm7, xmm1);
    
    // vpshufd xmm0,xmm0,0
    xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0x00);
    // vpshufd xmm1,xmm3,85
    xmm1 = _mm_shuffle_ps(xmm3, xmm3, 0x55);
    // vpshufd xmm2,xmm3,170
    xmm2 = _mm_shuffle_ps(xmm3, xmm3, 0xAA);
    // vpshufd xmm3,xmm3,255
    xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0xFF);
    
    // vmulps xmm0,xmm0,XMMWORD ptr [rdx+864]
    xmm0 = _mm_mul_ps(xmm0, _mm_load_ps(weights + 216));
    // vfmadd231ps xmm0,xmm1,XMMWORD ptr [rdx+864+16]
    xmm0 = _mm_fmadd_ps(xmm1, _mm_load_ps(weights + 220), xmm0);
    // vmulps xmm2,xmm2,XMMWORD ptr [rdx+864+32]
    xmm2 = _mm_mul_ps(xmm2, _mm_load_ps(weights + 224));
    // vfmadd231ps xmm2,xmm3,XMMWORD ptr [rdx+864+48]
    xmm2 = _mm_fmadd_ps(xmm3, _mm_load_ps(weights + 228), xmm2);
    
    // vpshufd xmm4,xmm7,0
    xmm4 = _mm_shuffle_ps(xmm7, xmm7, 0x00);
    // vpshufd xmm5,xmm7,85
    __m128 xmm5 = _mm_shuffle_ps(xmm7, xmm7, 0x55);
    // vpshufd xmm6,xmm7,170
    __m128 xmm6 = _mm_shuffle_ps(xmm7, xmm7, 0xAA);
    // vpshufd xmm7,xmm7,255
    xmm7 = _mm_shuffle_ps(xmm7, xmm7, 0xFF);
    
    // vmulps xmm4,xmm4,XMMWORD ptr [rdx+864+64]
    xmm4 = _mm_mul_ps(xmm4, _mm_load_ps(weights + 232));
    // vfmadd231ps xmm4,xmm5,XMMWORD ptr [rdx+864+80]
    xmm4 = _mm_fmadd_ps(xmm5, _mm_load_ps(weights + 236), xmm4);
    // vmulps xmm6,xmm6,XMMWORD ptr [rdx+864+96]
    xmm6 = _mm_mul_ps(xmm6, _mm_load_ps(weights + 240));
    // vfmadd231ps xmm6,xmm7,XMMWORD ptr [rdx+864+112]
    xmm6 = _mm_fmadd_ps(xmm7, _mm_load_ps(weights + 244), xmm6);
    
    // vaddps xmm0,xmm0,xmm2
    xmm0 = _mm_add_ps(xmm0, xmm2);
    // vaddps xmm4,xmm4,xmm6
    xmm4 = _mm_add_ps(xmm4, xmm6);
    // vaddps xmm0,xmm0,xmm4
    xmm0 = _mm_add_ps(xmm0, xmm4);
    // mov rcx,r8
    // vaddps xmm0,xmm0,XMMWORD ptr [rdx+864+128]
    xmm0 = _mm_add_ps(xmm0, _mm_load_ps(weights + 248));
    // vmovhlps xmm1,xmm1,xmm0
    xmm1 = _mm_movehl_ps(xmm1, xmm0);
    // vmaxps xmm0,xmm0,xmm1
    xmm0 = _mm_max_ps(xmm0, xmm1);
    // vpshuflw xmm1,xmm0,14
    xmm1 = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(xmm0), 0xE));
    // vcomiss xmm1,xmm0
    // jbe finish_1a
    // xor rax,rax
    if (_mm_comigt_ss(xmm1, xmm0)) {
        result = false;
    }
    
    // mov BYTE PTR[rcx],al
    *ptr_d = result;
    
    // vzeroupper
    _mm256_zeroupper();
}

extern "C"
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void computeNetwork0_i16_AVX2(const float* inputf_raw, const float* weightsf_raw, uint8_t* ptr_d) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int16_t* inputf = reinterpret_cast<const int16_t*>(inputf_raw);
    const int16_t* weightsf = reinterpret_cast<const int16_t*>(weightsf_raw);

    // vmovdqa ymm7,YMMWORD ptr [rcx]
    __m256i ymm7 = _mm256_load_si256((__m256i*)inputf);
    // vpmaddwd ymm0,ymm7,YMMWORD ptr [rdx]
    __m256i ymm0 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)weightsf));
    // vpmaddwd ymm1,ymm7,YMMWORD ptr [rdx+32]
    __m256i ymm1 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 16)));
    // vpmaddwd ymm2,ymm7,YMMWORD ptr [rdx+64]
    __m256i ymm2 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 32)));
    // vpmaddwd ymm3,ymm7,YMMWORD ptr [rdx+96]
    __m256i ymm3 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 48)));

    // vmovdqa ymm7,YMMWORD ptr [rcx+32]
    ymm7 = _mm256_load_si256((__m256i*)(inputf + 16));
    // vpmaddwd ymm4,ymm7,YMMWORD ptr [rdx+128]
    __m256i ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 64)));
    // vpmaddwd ymm5,ymm7,YMMWORD ptr [rdx+160]
    __m256i ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 80)));
    // vpmaddwd ymm6,ymm7,YMMWORD ptr [rdx+192]
    __m256i ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 96)));
    // vpmaddwd ymm7,ymm7,YMMWORD ptr [rdx+224]
    __m256i ymm7_temp = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 112)));
    // vpaddd ymm0,ymm0,ymm4
    ymm0 = _mm256_add_epi32(ymm0, ymm4);
    // vpaddd ymm1,ymm1,ymm5
    ymm1 = _mm256_add_epi32(ymm1, ymm5);
    // vpaddd ymm2,ymm2,ymm6
    ymm2 = _mm256_add_epi32(ymm2, ymm6);
    // vpaddd ymm3,ymm3,ymm7
    ymm3 = _mm256_add_epi32(ymm3, ymm7_temp);

    // vmovdqa ymm7,YMMWORD ptr [rcx+64]
    ymm7 = _mm256_load_si256((__m256i*)(inputf + 32));
    // vpmaddwd ymm4,ymm7,YMMWORD ptr [rdx+256]
    ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 128)));
    // vpmaddwd ymm5,ymm7,YMMWORD ptr [rdx+288]
    ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 144)));
    // vpmaddwd ymm6,ymm7,YMMWORD ptr [rdx+320]
    ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 160)));
    // vpmaddwd ymm7,ymm7,YMMWORD ptr [rdx+352]
    ymm7_temp = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(weightsf + 176)));
    // vpaddd ymm0,ymm0,ymm4
    ymm0 = _mm256_add_epi32(ymm0, ymm4);
    // vpaddd ymm1,ymm1,ymm5
    ymm1 = _mm256_add_epi32(ymm1, ymm5);
    // vpaddd ymm2,ymm2,ymm6
    ymm2 = _mm256_add_epi32(ymm2, ymm6);
    // vpaddd ymm3,ymm3,ymm7
    ymm3 = _mm256_add_epi32(ymm3, ymm7_temp);

    // vpunpckhqdq ymm4,ymm0,ymm1
    ymm4 = _mm256_unpackhi_epi64(ymm0, ymm1);
    // vpunpckhqdq ymm5,ymm2,ymm3
    ymm5 = _mm256_unpackhi_epi64(ymm2, ymm3);
    // vpunpcklqdq ymm0,ymm0,ymm1
    ymm0 = _mm256_unpacklo_epi64(ymm0, ymm1);
    // vpunpcklqdq ymm2,ymm2,ymm3
    ymm2 = _mm256_unpacklo_epi64(ymm2, ymm3);
    // vpaddd ymm0,ymm0,ymm4
    ymm0 = _mm256_add_epi32(ymm0, ymm4);
    // vpaddd ymm2,ymm2,ymm5
    ymm2 = _mm256_add_epi32(ymm2, ymm5);

    // vextracti128 xmm4,ymm0,1
    __m128i xmm4 = _mm256_extracti128_si256(ymm0, 1);
    // vextracti128 xmm5,ymm2,1
    __m128i xmm5 = _mm256_extracti128_si256(ymm2, 1);
    // vpaddd xmm0,xmm0,xmm4
    __m128i xmm0 = _mm_add_epi32(_mm256_castsi256_si128(ymm0), xmm4);
    // vpaddd xmm2,xmm2,xmm5
    __m128i xmm2 = _mm_add_epi32(_mm256_castsi256_si128(ymm2), xmm5);

    // vshufps xmm6,xmm0,xmm2,221
    __m128i xmm6 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2), 0xDD));
    // vshufps xmm0,xmm0,xmm2,136
    xmm0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2), 0x88));

    // vpaddd xmm0,xmm0,xmm6
    xmm0 = _mm_add_epi32(xmm0, xmm6);
    // vcvtdq2ps xmm0,xmm0
    __m128 xmm0_ps = _mm_cvtepi32_ps(xmm0);
    // vmulps xmm0,xmm0,XMMWORD ptr [rdx+384]
    xmm0_ps = _mm_mul_ps(xmm0_ps, _mm_load_ps((float*)(weightsf + 192)));
    // vaddps xmm0,xmm0,XMMWORD ptr [rdx+400]
    xmm0_ps = _mm_add_ps(xmm0_ps, _mm_load_ps((float*)(weightsf + 200)));

    // vmovaps xmm1,xmm0
    __m128 xmm1 = xmm0_ps;
    // vandps xmm0,xmm0,XMMWORD ptr sign_bits_f_zero_l
    xmm0_ps = _mm_and_ps(xmm0_ps, sign_bits_f_zero_l);
    // vaddps xmm0,xmm0,XMMWORD ptr ones_f
    xmm0_ps = _mm_add_ps(xmm0_ps, ones_f);
    // vrcpps xmm0,xmm0
    xmm0_ps = _mm_rcp_ps(xmm0_ps);
    // vmulps xmm0,xmm0,xmm1
    xmm0_ps = _mm_mul_ps(xmm0_ps, xmm1);

    // vpshufd xmm1,xmm0,0
    xmm1 = _mm_shuffle_ps(xmm0_ps, xmm0_ps, 0);
    // vpshufd xmm2,xmm0,85
    __m128 xmm2_ps = _mm_shuffle_ps(xmm0_ps, xmm0_ps, 0x55);
    // vpshufd xmm3,xmm0,170
    __m128 xmm3 = _mm_shuffle_ps(xmm0_ps, xmm0_ps, 0xAA);
    // vpshufd xmm4,xmm0,255
    __m128 xmm4_ps = _mm_shuffle_ps(xmm0_ps, xmm0_ps, 0xFF);

    // vmulps xmm1,xmm1,XMMWORD ptr [rdx+416]
    xmm1 = _mm_mul_ps(xmm1, _mm_load_ps((float*)(weightsf + 208)));
    // vmulps xmm2,xmm2,XMMWORD ptr [rdx+416+16]
    xmm2_ps = _mm_mul_ps(xmm2_ps, _mm_load_ps((float*)(weightsf + 216)));
    // vmulps xmm3,xmm3,XMMWORD ptr [rdx+416+32]
    xmm3 = _mm_mul_ps(xmm3, _mm_load_ps((float*)(weightsf + 224)));
    // vmulps xmm4,xmm4,XMMWORD ptr [rdx+416+48]
    xmm4_ps = _mm_mul_ps(xmm4_ps, _mm_load_ps((float*)(weightsf + 232)));

    // vaddps xmm1,xmm1,xmm2
    xmm1 = _mm_add_ps(xmm1, xmm2_ps);
    // vaddps xmm3,xmm3,xmm4
    xmm3 = _mm_add_ps(xmm3, xmm4_ps);
    // vaddps xmm1,xmm1,xmm3
    xmm1 = _mm_add_ps(xmm1, xmm3);
    // mov rcx,r8
    // vaddps xmm1,xmm1,XMMWORD ptr [rdx+416+64]
    xmm1 = _mm_add_ps(xmm1, _mm_load_ps((float*)(weightsf + 240)));

    // 第2層のElliott活性化
    __m128 xmm7 = xmm1;
    xmm1 = _mm_and_ps(xmm1, sign_bits_f);
    __m128 xmm3_input = xmm0_ps;
    xmm1 = _mm_add_ps(xmm1, ones_f);
    xmm1 = _mm_rcp_ps(xmm1);
    xmm7 = _mm_mul_ps(xmm7, xmm1);

    // 最終層: 第1層4値と第2層4値から4出力を計算する
    __m128 xmm0_l0 = _mm_shuffle_ps(xmm0_ps, xmm0_ps, 0x00);
    __m128 xmm1_l1 = _mm_shuffle_ps(xmm3_input, xmm3_input, 0x55);
    __m128 xmm2_l2 = _mm_shuffle_ps(xmm3_input, xmm3_input, 0xAA);
    __m128 xmm3_l3 = _mm_shuffle_ps(xmm3_input, xmm3_input, 0xFF);

    xmm0_l0 = _mm_mul_ps(xmm0_l0, _mm_load_ps((const float*)(weightsf + 248)));
    xmm1_l1 = _mm_mul_ps(xmm1_l1, _mm_load_ps((const float*)(weightsf + 256)));
    xmm2_l2 = _mm_mul_ps(xmm2_l2, _mm_load_ps((const float*)(weightsf + 264)));
    xmm3_l3 = _mm_mul_ps(xmm3_l3, _mm_load_ps((const float*)(weightsf + 272)));

    __m128 xmm4_l4 = _mm_shuffle_ps(xmm7, xmm7, 0x00);
    __m128 xmm5_l5 = _mm_shuffle_ps(xmm7, xmm7, 0x55);
    __m128 xmm6_l6 = _mm_shuffle_ps(xmm7, xmm7, 0xAA);
    __m128 xmm7_l7 = _mm_shuffle_ps(xmm7, xmm7, 0xFF);

    xmm4_l4 = _mm_mul_ps(xmm4_l4, _mm_load_ps((const float*)(weightsf + 280)));
    xmm5_l5 = _mm_mul_ps(xmm5_l5, _mm_load_ps((const float*)(weightsf + 288)));
    xmm6_l6 = _mm_mul_ps(xmm6_l6, _mm_load_ps((const float*)(weightsf + 296)));
    xmm7_l7 = _mm_mul_ps(xmm7_l7, _mm_load_ps((const float*)(weightsf + 304)));

    xmm0_l0 = _mm_add_ps(xmm0_l0, xmm1_l1);
    xmm2_l2 = _mm_add_ps(xmm2_l2, xmm3_l3);
    xmm4_l4 = _mm_add_ps(xmm4_l4, xmm5_l5);
    xmm6_l6 = _mm_add_ps(xmm6_l6, xmm7_l7);
    xmm0_l0 = _mm_add_ps(xmm0_l0, xmm2_l2);
    xmm4_l4 = _mm_add_ps(xmm4_l4, xmm6_l6);
    xmm0_l0 = _mm_add_ps(xmm0_l0, xmm4_l4);
    xmm0_l0 = _mm_add_ps(xmm0_l0, _mm_load_ps((const float*)(weightsf + 312)));

    // SIMD用配置の出力laneは0,2,1,3の順。max(out[0],out[1]) >= max(out[2],out[3])ならprescreenerを通す。
    __m128 high_pair = _mm_movehl_ps(xmm0_l0, xmm0_l0);
    __m128 pair_max = _mm_max_ps(xmm0_l0, high_pair);
    __m128 second = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(pair_max), 0x0E));
    *ptr_d = _mm_comigt_ss(second, pair_max) ? 0 : 1;

    // vzeroupper
    _mm256_zeroupper();

}

// computeNetwork0new_AVX2 proc datai:dword,weights:dword,ptr_d:dword
// datai = rcx
// weights = rdx
// ptr_d = r8
extern "C" void computeNetwork0new_AVX2(const float* datai_raw, const float* weights, uint8_t* ptr_d) {
    const int16_t* datai = reinterpret_cast<const int16_t*>(datai_raw);
    // sub rsp,32
    // .allocstack 32
    // vmovdqu XMMWORD ptr[rsp],xmm6
    // .savexmm128 xmm6,0
    // vmovdqu XMMWORD ptr[rsp+16],xmm7
    // .savexmm128 xmm7,16
    // .endprolog

    // mov rax,rdx
    const float* rax = weights;

    // vmovdqa ymm7,YMMWORD ptr [rcx]
    __m256i ymm7 = _mm256_load_si256((__m256i*)datai);
    // vpmaddwd ymm0,ymm7,YMMWORD ptr [rax]
    __m256i ymm0 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)rax));
    // vpmaddwd ymm1,ymm7,YMMWORD ptr [rax+32]
    __m256i ymm1 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 8)));
    // vpmaddwd ymm2,ymm7,YMMWORD ptr [rax+64]
    __m256i ymm2 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 16)));
    // vpmaddwd ymm3,ymm7,YMMWORD ptr [rax+96]
    __m256i ymm3 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 24)));

    // vmovdqa ymm7,YMMWORD ptr [rcx+32]
    ymm7 = _mm256_load_si256((__m256i*)(datai + 16));
    // vpmaddwd ymm4,ymm7,YMMWORD ptr [rax+128]
    __m256i ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 32)));
    // vpmaddwd ymm5,ymm7,YMMWORD ptr [rax+160]
    __m256i ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 40)));
    // vpmaddwd ymm6,ymm7,YMMWORD ptr [rax+192]
    __m256i ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 48)));
    // vpmaddwd ymm7,ymm7,YMMWORD ptr [rax+224]
    __m256i ymm7_temp = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 56)));
    // vpaddd ymm0,ymm0,ymm4
    ymm0 = _mm256_add_epi32(ymm0, ymm4);
    // vpaddd ymm1,ymm1,ymm5
    ymm1 = _mm256_add_epi32(ymm1, ymm5);
    // vpaddd ymm2,ymm2,ymm6
    ymm2 = _mm256_add_epi32(ymm2, ymm6);
    // vpaddd ymm3,ymm3,ymm7
    ymm3 = _mm256_add_epi32(ymm3, ymm7_temp);

    // vmovdqa ymm7,YMMWORD ptr [rcx+64]
    ymm7 = _mm256_load_si256((__m256i*)(datai + 32));
    // vpmaddwd ymm4,ymm7,YMMWORD ptr [rax+256]
    ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 64)));
    // vpmaddwd ymm5,ymm7,YMMWORD ptr [rax+288]
    ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 72)));
    // vpmaddwd ymm6,ymm7,YMMWORD ptr [rax+320]
    ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 80)));
    // vpmaddwd ymm7,ymm7,YMMWORD ptr [rax+352]
    ymm7_temp = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rax + 88)));
    // vpaddd ymm0,ymm0,ymm4
    ymm0 = _mm256_add_epi32(ymm0, ymm4);
    // vpaddd ymm1,ymm1,ymm5
    ymm1 = _mm256_add_epi32(ymm1, ymm5);
    // vpaddd ymm2,ymm2,ymm6
    ymm2 = _mm256_add_epi32(ymm2, ymm6);
    // vpaddd ymm3,ymm3,ymm7
    ymm3 = _mm256_add_epi32(ymm3, ymm7_temp);

    // 4番目の16入力と各ニューロンの重みを積和する
    ymm7 = _mm256_load_si256((const __m256i*)(datai + 48));
    ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((const __m256i*)(rax + 96)));
    ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((const __m256i*)(rax + 104)));
    ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((const __m256i*)(rax + 112)));
    ymm7_temp = _mm256_madd_epi16(ymm7, _mm256_load_si256((const __m256i*)(rax + 120)));
    ymm0 = _mm256_add_epi32(ymm0, ymm4);
    ymm1 = _mm256_add_epi32(ymm1, ymm5);
    ymm2 = _mm256_add_epi32(ymm2, ymm6);
    ymm3 = _mm256_add_epi32(ymm3, ymm7_temp);

    // vpunpckhqdq ymm4,ymm0,ymm1
    ymm4 = _mm256_unpackhi_epi64(ymm0, ymm1);
    // vpunpckhqdq ymm5,ymm2,ymm3
    ymm5 = _mm256_unpackhi_epi64(ymm2, ymm3);
    // vpunpcklqdq ymm0,ymm0,ymm1
    ymm0 = _mm256_unpacklo_epi64(ymm0, ymm1);
    // vpunpcklqdq ymm2,ymm2,ymm3
    ymm2 = _mm256_unpacklo_epi64(ymm2, ymm3);
    // vpaddd ymm0,ymm0,ymm4
    ymm0 = _mm256_add_epi32(ymm0, ymm4);
    // vpaddd ymm2,ymm2,ymm5
    ymm2 = _mm256_add_epi32(ymm2, ymm5);

    // vextracti128 xmm4,ymm0,1
    __m128i xmm4 = _mm256_extracti128_si256(ymm0, 1);
    // vextracti128 xmm5,ymm2,1
    __m128i xmm5 = _mm256_extracti128_si256(ymm2, 1);
    // vpaddd xmm0,xmm0,xmm4
    __m128i xmm0 = _mm_add_epi32(_mm256_castsi256_si128(ymm0), xmm4);
    // vpaddd xmm2,xmm2,xmm5
    __m128i xmm2 = _mm_add_epi32(_mm256_castsi256_si128(ymm2), xmm5);

    // vshufps xmm6,xmm0,xmm2,221
    __m128i xmm6 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2), 0xDD));
    // vshufps xmm0,xmm0,xmm2,136
    xmm0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2), 0x88));

    // vpaddd xmm0,xmm0,xmm6
    xmm0 = _mm_add_epi32(xmm0, xmm6);
    // vcvtdq2ps xmm0,xmm0
    __m128 xmm0_ps = _mm_cvtepi32_ps(xmm0);
    // vmulps xmm0,xmm0,XMMWORD ptr [rax+512]
    xmm0_ps = _mm_mul_ps(xmm0_ps, _mm_load_ps(rax + 128));
    // vaddps xmm0,xmm0,XMMWORD ptr [rax+528]
    xmm0_ps = _mm_add_ps(xmm0_ps, _mm_load_ps(rax + 132));
    // vmovaps xmm1,xmm0
    __m128 xmm1 = xmm0_ps;
    // vandps xmm0,xmm0,XMMWORD ptr sign_bits_f
    xmm0_ps = _mm_and_ps(xmm0_ps, sign_bits_f);
    // vaddps xmm0,xmm0,XMMWORD ptr ones_f
    xmm0_ps = _mm_add_ps(xmm0_ps, ones_f);
    // vrcpps xmm0,xmm0
    xmm0_ps = _mm_rcp_ps(xmm0_ps);
    // vmulps xmm0,xmm0,xmm1
    xmm0_ps = _mm_mul_ps(xmm0_ps, xmm1);

    // vpshufd xmm1,xmm0,0
    xmm1 = _mm_shuffle_ps(xmm0_ps, xmm0_ps, 0);
    // vpshufd xmm2,xmm0,85
    __m128 xmm2_ps = _mm_shuffle_ps(xmm0_ps, xmm0_ps, 0x55);
    // vpshufd xmm3,xmm0,170
    __m128 xmm3 = _mm_shuffle_ps(xmm0_ps, xmm0_ps, 0xAA);
    // vpshufd xmm4,xmm0,255
    __m128 xmm4_ps = _mm_shuffle_ps(xmm0_ps, xmm0_ps, 0xFF);

    // vmulps xmm1,xmm1,XMMWORD ptr [rax+544]
    xmm1 = _mm_mul_ps(xmm1, _mm_load_ps(rax + 136));
    // vmulps xmm2,xmm2,XMMWORD ptr [rax+560]
    xmm2_ps = _mm_mul_ps(xmm2_ps, _mm_load_ps(rax + 140));
    // vmulps xmm3,xmm3,XMMWORD ptr [rax+576]
    xmm3 = _mm_mul_ps(xmm3, _mm_load_ps(rax + 144));
    // vmulps xmm4,xmm4,XMMWORD ptr [rax+592]
    xmm4_ps = _mm_mul_ps(xmm4_ps, _mm_load_ps(rax + 148));

    // vpxor xmm0,xmm0,xmm0
    xmm0_ps = _mm_setzero_ps();
    // vaddps xmm1,xmm1,xmm2
    xmm1 = _mm_add_ps(xmm1, xmm2_ps);
    // vaddps xmm3,xmm3,xmm4
    xmm3 = _mm_add_ps(xmm3, xmm4_ps);
    // vaddps xmm1,xmm1,xmm3
    xmm1 = _mm_add_ps(xmm1, xmm3);
    // mov rcx,r8
    // vaddps xmm1,xmm1,XMMWORD ptr [rax+608]
    xmm1 = _mm_add_ps(xmm1, _mm_load_ps(rax + 152));
    // vcmpps xmm1,xmm1,xmm0,1
    xmm1 = _mm_cmplt_ps(xmm1, xmm0_ps);
    // vpackssdw xmm1,xmm1,xmm0
    xmm1 = _mm_castsi128_ps(_mm_packs_epi32(_mm_castps_si128(xmm1), _mm_castps_si128(xmm0_ps)));
    // vpacksswb xmm1,xmm1,xmm0
    xmm1 = _mm_castsi128_ps(_mm_packs_epi16(_mm_castps_si128(xmm1), _mm_castps_si128(xmm0_ps)));

    // vmovd eax,xmm1
    uint32_t eax = _mm_cvtsi128_si32(_mm_castps_si128(xmm1));
    // xor eax,0FFFFFFFFh
    eax ^= 0xFFFFFFFF;
    // and eax,001010101h
    eax &= 0x01010101;
    // mov [rcx],eax
    std::memcpy(ptr_d, &eax, sizeof(eax));

    // vzeroupper
    _mm256_zeroupper();
}

// uc2f48_AVX2 proc ptr_t:dword,pitch:dword,ptr_p:dword
// ptr_t = rcx
// pitch = edx
// ptr_p = r8
extern "C" void uc2f48_AVX2(const uint8_t* ptr_t, int pitch, float* ptr_p) {
    // .endprolog

    // mov rax,rcx
    const uint8_t* rax = ptr_t;
    // movsxd rcx,edx
    int64_t rcx = pitch;
    // vpxor ymm4,ymm4,ymm4
    __m256i ymm4 = _mm256_setzero_si256();

    // test rax,15
    __m128i xmm0, xmm2;
    if ((reinterpret_cast<uintptr_t>(rax) & 15) == 0) {
        // vmovdqa xmm0,XMMWORD PTR[rax]
        xmm0 = _mm_load_si128((__m128i*)rax);
        // vmovdqa xmm2,XMMWORD PTR[rax+rcx*2]
        xmm2 = _mm_load_si128((__m128i*)(rax + rcx * 2));
        // vmovhlps xmm1,xmm4,xmm0
        __m128i xmm1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm0)));
        // vmovhlps xmm3,xmm4,xmm2
        __m128i xmm3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm2)));
        // vinserti128 ymm0,ymm0,xmm1,1
        __m256i ymm0 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm0), xmm1, 1);
        // vinserti128 ymm2,ymm2,xmm3,1
        __m256i ymm2 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm2), xmm3, 1);

        // vpunpcklbw ymm1,ymm0,ymm4
        __m256i ymm1 = _mm256_unpacklo_epi8(ymm0, ymm4);
        // vpunpcklbw ymm3,ymm2,ymm4
        __m256i ymm3 = _mm256_unpacklo_epi8(ymm2, ymm4);

        // vpunpcklwd ymm0,ymm1,ymm4
        ymm0 = _mm256_unpacklo_epi16(ymm1, ymm4);
        // vpunpcklwd ymm2,ymm3,ymm4
        ymm2 = _mm256_unpacklo_epi16(ymm3, ymm4);
        // vpunpckhwd ymm1,ymm1,ymm4
        ymm1 = _mm256_unpackhi_epi16(ymm1, ymm4);
        // vpunpckhwd ymm3,ymm3,ymm4
        ymm3 = _mm256_unpackhi_epi16(ymm3, ymm4);
        // lea rax,[rax+rcx*4]
        rax += rcx * 4;

        // vcvtdq2ps ymm0,ymm0
        __m256 ymm0_ps = _mm256_cvtepi32_ps(ymm0);
        // vcvtdq2ps xmm1,xmm1
        __m128 xmm1_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm1));
        // vcvtdq2ps ymm2,ymm2
        __m256 ymm2_ps = _mm256_cvtepi32_ps(ymm2);
        // vcvtdq2ps xmm3,xmm3
        __m128 xmm3_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm3));

        // vmovaps XMMWORD ptr[r8],xmm0
        _mm_store_ps(ptr_p, _mm256_castps256_ps128(ymm0_ps));
        // vextractf128 xmm5,ymm0,1
        __m128 xmm5 = _mm256_extractf128_ps(ymm0_ps, 1);
        // vmovaps XMMWORD ptr[r8+16],xmm1
        _mm_store_ps(ptr_p + 4, xmm1_ps);
        // vmovaps XMMWORD ptr[r8+32],xmm5
        _mm_store_ps(ptr_p + 8, xmm5);
        // vmovaps XMMWORD ptr[r8+48],xmm2
        _mm_store_ps(ptr_p + 12, _mm256_castps256_ps128(ymm2_ps));
        // vextractf128 xmm5,ymm2,1
        xmm5 = _mm256_extractf128_ps(ymm2_ps, 1);
        // vmovaps XMMWORD ptr[r8+64],xmm3
        _mm_store_ps(ptr_p + 16, xmm3_ps);
        // vmovaps XMMWORD ptr[r8+80],xmm5
        _mm_store_ps(ptr_p + 20, xmm5);

        // vmovdqu xmm0,XMMWORD PTR[rax]
        xmm0 = _mm_loadu_si128((__m128i*)rax);
        // vmovdqu xmm2,XMMWORD PTR[rax+rcx*2]
        xmm2 = _mm_loadu_si128((__m128i*)(rax + rcx * 2));
    } else {
        // unaligned_1:
        // vmovdqu xmm0,XMMWORD PTR[rax]
        xmm0 = _mm_loadu_si128((__m128i*)rax);
        // vmovdqu xmm2,XMMWORD PTR[rax+rcx*2]
        xmm2 = _mm_loadu_si128((__m128i*)(rax + rcx * 2));
        // vmovhlps xmm1,xmm4,xmm0
        __m128i xmm1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm0)));
        // vmovhlps xmm3,xmm4,xmm2
        __m128i xmm3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm2)));
        // vinserti128 ymm0,ymm0,xmm1,1
        __m256i ymm0 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm0), xmm1, 1);
        // vinserti128 ymm2,ymm2,xmm3,1
        __m256i ymm2 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm2), xmm3, 1);

        // vpunpcklbw ymm1,ymm0,ymm4
        __m256i ymm1 = _mm256_unpacklo_epi8(ymm0, ymm4);
        // vpunpcklbw ymm3,ymm2,ymm4
        __m256i ymm3 = _mm256_unpacklo_epi8(ymm2, ymm4);

        // vpunpcklwd ymm0,ymm1,ymm4
        ymm0 = _mm256_unpacklo_epi16(ymm1, ymm4);
        // vpunpcklwd ymm2,ymm3,ymm4
        ymm2 = _mm256_unpacklo_epi16(ymm3, ymm4);
        // vpunpckhwd ymm1,ymm1,ymm4
        ymm1 = _mm256_unpackhi_epi16(ymm1, ymm4);
        // vpunpckhwd ymm3,ymm3,ymm4
        ymm3 = _mm256_unpackhi_epi16(ymm3, ymm4);
        // lea rax,[rax+rcx*4]
        rax += rcx * 4;

        // vcvtdq2ps ymm0,ymm0
        __m256 ymm0_ps = _mm256_cvtepi32_ps(ymm0);
        // vcvtdq2ps xmm1,xmm1
        __m128 xmm1_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm1));
        // vcvtdq2ps ymm2,ymm2
        __m256 ymm2_ps = _mm256_cvtepi32_ps(ymm2);
        // vcvtdq2ps xmm3,xmm3
        __m128 xmm3_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm3));

        // vmovaps XMMWORD ptr[r8],xmm0
        _mm_store_ps(ptr_p, _mm256_castps256_ps128(ymm0_ps));
        // vextractf128 xmm5,ymm0,1
        __m128 xmm5 = _mm256_extractf128_ps(ymm0_ps, 1);
        // vmovaps XMMWORD ptr[r8+16],xmm1
        _mm_store_ps(ptr_p + 4, xmm1_ps);
        // vmovaps XMMWORD ptr[r8+32],xmm5
        _mm_store_ps(ptr_p + 8, xmm5);
        // vmovaps XMMWORD ptr[r8+48],xmm2
        _mm_store_ps(ptr_p + 12, _mm256_castps256_ps128(ymm2_ps));
        // vextractf128 xmm5,ymm2,1
        xmm5 = _mm256_extractf128_ps(ymm2_ps, 1);
        // vmovaps XMMWORD ptr[r8+64],xmm3
        _mm_store_ps(ptr_p + 16, xmm3_ps);
        // vmovaps XMMWORD ptr[r8+80],xmm5
        _mm_store_ps(ptr_p + 20, xmm5);

        // vmovdqu xmm0,XMMWORD PTR[rax]
        xmm0 = _mm_loadu_si128((__m128i*)rax);
        // vmovdqu xmm2,XMMWORD PTR[rax+rcx*2]
        xmm2 = _mm_loadu_si128((__m128i*)(rax + rcx * 2));
    }

    // suite_1:
    // vmovhlps xmm1,xmm4,xmm0
    __m128i xmm1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm0)));
    // vmovhlps xmm3,xmm4,xmm2
    __m128i xmm3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm2)));
    // vinserti128 ymm0,ymm0,xmm1,1
    __m256i ymm0 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm0), xmm1, 1);
    // vinserti128 ymm2,ymm2,xmm3,1
    __m256i ymm2 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm2), xmm3, 1);

    // vpunpcklbw ymm1,ymm0,ymm4
    __m256i ymm1 = _mm256_unpacklo_epi8(ymm0, ymm4);
    // vpunpcklbw ymm3,ymm2,ymm4
    __m256i ymm3 = _mm256_unpacklo_epi8(ymm2, ymm4);
    // vpunpcklwd ymm0,ymm1,ymm4
    ymm0 = _mm256_unpacklo_epi16(ymm1, ymm4);
    // vpunpcklwd ymm2,ymm3,ymm4
    ymm2 = _mm256_unpacklo_epi16(ymm3, ymm4);
    // vpunpckhwd ymm1,ymm1,ymm4
    ymm1 = _mm256_unpackhi_epi16(ymm1, ymm4);
    // vpunpckhwd ymm3,ymm3,ymm4
    ymm3 = _mm256_unpackhi_epi16(ymm3, ymm4);

    // vcvtdq2ps ymm0,ymm0
    __m256 ymm0_ps = _mm256_cvtepi32_ps(ymm0);
    // vcvtdq2ps xmm1,xmm1
    __m128 xmm1_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm1));
    // vcvtdq2ps ymm2,ymm2
    __m256 ymm2_ps = _mm256_cvtepi32_ps(ymm2);
    // vcvtdq2ps xmm3,xmm3
    __m128 xmm3_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm3));

    // vmovaps XMMWORD ptr[r8+96],xmm0
    _mm_store_ps(ptr_p + 24, _mm256_castps256_ps128(ymm0_ps));
    // vextractf128 xmm5,ymm0,1
    __m128 xmm5 = _mm256_extractf128_ps(ymm0_ps, 1);
    // vmovaps XMMWORD ptr[r8+112],xmm1
    _mm_store_ps(ptr_p + 28, xmm1_ps);
    // vmovaps XMMWORD ptr[r8+128],xmm5
    _mm_store_ps(ptr_p + 32, xmm5);
    // vmovaps XMMWORD ptr[r8+144],xmm2
    _mm_store_ps(ptr_p + 36, _mm256_castps256_ps128(ymm2_ps));
    // vextractf128 xmm5,ymm2,1
    xmm5 = _mm256_extractf128_ps(ymm2_ps, 1);
    // vmovaps XMMWORD ptr[r8+160],xmm3
    _mm_store_ps(ptr_p + 40, xmm3_ps);
    // vmovaps XMMWORD ptr[r8+176],xmm5
    _mm_store_ps(ptr_p + 44, xmm5);

    // vzeroupper
    _mm256_zeroupper();
}

// uc2f48_AVX2_16 proc ptr_t:dword,pitch:dword,ptr_p:dword
// ptr_t = rcx
// pitch = edx
// ptr_p = r8
extern "C" void uc2f48_AVX2_16(const uint8_t* ptr_t, int pitch, float* ptr_p) {
    // .endprolog

    // mov rax,rcx
    const uint8_t* rax = ptr_t;
    // movsxd rcx,edx
    int64_t rcx = pitch;
    // vpxor ymm4,ymm4,ymm4
    __m256i ymm4 = _mm256_setzero_si256();

    // test rax,31
    __m256i ymm1, ymm3;
    if ((reinterpret_cast<uintptr_t>(rax) & 31) == 0) {
        // vmovdqa ymm1,YMMWORD ptr[rax]
        ymm1 = _mm256_load_si256((__m256i*)rax);
        // vmovdqa ymm3,YMMWORD ptr[rax+rcx*2]
        ymm3 = _mm256_load_si256((__m256i*)(rax + rcx * 2));
        // vpunpcklwd ymm0,ymm1,ymm4
        __m256i ymm0 = _mm256_unpacklo_epi16(ymm1, ymm4);
        // vpunpcklwd ymm2,ymm3,ymm4
        __m256i ymm2 = _mm256_unpacklo_epi16(ymm3, ymm4);
        // vpunpckhwd ymm1,ymm1,ymm4
        ymm1 = _mm256_unpackhi_epi16(ymm1, ymm4);
        // vpunpckhwd ymm3,ymm3,ymm4
        ymm3 = _mm256_unpackhi_epi16(ymm3, ymm4);
        // lea rax,[rax+rcx*4]
        rax += rcx * 4;

        // vcvtdq2ps ymm0,ymm0
        __m256 ymm0_ps = _mm256_cvtepi32_ps(ymm0);
        // vcvtdq2ps xmm1,xmm1
        __m128 xmm1_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm1));
        // vcvtdq2ps ymm2,ymm2
        __m256 ymm2_ps = _mm256_cvtepi32_ps(ymm2);
        // vcvtdq2ps xmm3,xmm3
        __m128 xmm3_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm3));

        // vmovaps XMMWORD ptr[r8],xmm0
        _mm_store_ps(ptr_p, _mm256_castps256_ps128(ymm0_ps));
        // vextractf128 xmm5,ymm0,1
        __m128 xmm5 = _mm256_extractf128_ps(ymm0_ps, 1);
        // vmovaps XMMWORD ptr[r8+16],xmm1
        _mm_store_ps(ptr_p + 4, xmm1_ps);
        // vmovaps XMMWORD ptr[r8+32],xmm5
        _mm_store_ps(ptr_p + 8, xmm5);
        // vmovaps XMMWORD ptr[r8+48],xmm2
        _mm_store_ps(ptr_p + 12, _mm256_castps256_ps128(ymm2_ps));
        // vextractf128 xmm5,ymm2,1
        xmm5 = _mm256_extractf128_ps(ymm2_ps, 1);
        // vmovaps XMMWORD ptr[r8+64],xmm3
        _mm_store_ps(ptr_p + 16, xmm3_ps);
        // vmovaps XMMWORD ptr[r8+80],xmm5
        _mm_store_ps(ptr_p + 20, xmm5);

        // vmovdqa ymm1,YMMWORD ptr[rax]
        ymm1 = _mm256_load_si256((__m256i*)rax);
        // vmovdqa ymm3,YMMWORD ptr[rax+rcx*2]
        ymm3 = _mm256_load_si256((__m256i*)(rax + rcx * 2));
    } else {
        // unaligned_2:
        // vmovdqu ymm1,YMMWORD ptr[rax]
        ymm1 = _mm256_loadu_si256((__m256i*)rax);
        // vmovdqu ymm3,YMMWORD ptr[rax+rcx*2]
        ymm3 = _mm256_loadu_si256((__m256i*)(rax + rcx * 2));
        // vpunpcklwd ymm0,ymm1,ymm4
        __m256i ymm0 = _mm256_unpacklo_epi16(ymm1, ymm4);
        // vpunpcklwd ymm2,ymm3,ymm4
        __m256i ymm2 = _mm256_unpacklo_epi16(ymm3, ymm4);
        // vpunpckhwd ymm1,ymm1,ymm4
        ymm1 = _mm256_unpackhi_epi16(ymm1, ymm4);
        // vpunpckhwd ymm3,ymm3,ymm4
        ymm3 = _mm256_unpackhi_epi16(ymm3, ymm4);
        // lea rax,[rax+rcx*4]
        rax += rcx * 4;

        // vcvtdq2ps ymm0,ymm0
        __m256 ymm0_ps = _mm256_cvtepi32_ps(ymm0);
        // vcvtdq2ps xmm1,xmm1
        __m128 xmm1_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm1));
        // vcvtdq2ps ymm2,ymm2
        __m256 ymm2_ps = _mm256_cvtepi32_ps(ymm2);
        // vcvtdq2ps xmm3,xmm3
        __m128 xmm3_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm3));

        // vmovaps XMMWORD ptr[r8],xmm0
        _mm_store_ps(ptr_p, _mm256_castps256_ps128(ymm0_ps));
        // vextractf128 xmm5,ymm0,1
        __m128 xmm5 = _mm256_extractf128_ps(ymm0_ps, 1);
        // vmovaps XMMWORD ptr[r8+16],xmm1
        _mm_store_ps(ptr_p + 4, xmm1_ps);
        // vmovaps XMMWORD ptr[r8+32],xmm5
        _mm_store_ps(ptr_p + 8, xmm5);
        // vmovaps XMMWORD ptr[r8+48],xmm2
        _mm_store_ps(ptr_p + 12, _mm256_castps256_ps128(ymm2_ps));
        // vextractf128 xmm5,ymm2,1
        xmm5 = _mm256_extractf128_ps(ymm2_ps, 1);
        // vmovaps XMMWORD ptr[r8+64],xmm3
        _mm_store_ps(ptr_p + 16, xmm3_ps);
        // vmovaps XMMWORD ptr[r8+80],xmm5
        _mm_store_ps(ptr_p + 20, xmm5);

        // vmovdqu ymm1,YMMWORD ptr[rax]
        ymm1 = _mm256_loadu_si256((__m256i*)rax);
        // vmovdqu ymm3,YMMWORD ptr[rax+rcx*2]
        ymm3 = _mm256_loadu_si256((__m256i*)(rax + rcx * 2));
    }

    // suite_2:
    // vpunpcklwd ymm0,ymm1,ymm4
    __m256i ymm0 = _mm256_unpacklo_epi16(ymm1, ymm4);
    // vpunpcklwd ymm2,ymm3,ymm4
    __m256i ymm2 = _mm256_unpacklo_epi16(ymm3, ymm4);
    // vpunpckhwd ymm1,ymm1,ymm4
    ymm1 = _mm256_unpackhi_epi16(ymm1, ymm4);
    // vpunpckhwd ymm3,ymm3,ymm4
    ymm3 = _mm256_unpackhi_epi16(ymm3, ymm4);

    // vcvtdq2ps ymm0,ymm0
    __m256 ymm0_ps = _mm256_cvtepi32_ps(ymm0);
    // vcvtdq2ps xmm1,xmm1
    __m128 xmm1_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm1));
    // vcvtdq2ps ymm2,ymm2
    __m256 ymm2_ps = _mm256_cvtepi32_ps(ymm2);
    // vcvtdq2ps xmm3,xmm3
    __m128 xmm3_ps = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm3));

    // vmovaps XMMWORD ptr[r8+96],xmm0
    _mm_store_ps(ptr_p + 24, _mm256_castps256_ps128(ymm0_ps));
    // vextractf128 xmm5,ymm0,1
    __m128 xmm5 = _mm256_extractf128_ps(ymm0_ps, 1);
    // vmovaps XMMWORD ptr[r8+112],xmm1
    _mm_store_ps(ptr_p + 28, xmm1_ps);
    // vmovaps XMMWORD ptr[r8+128],xmm5
    _mm_store_ps(ptr_p + 32, xmm5);
    // vmovaps XMMWORD ptr[r8+144],xmm2
    _mm_store_ps(ptr_p + 36, _mm256_castps256_ps128(ymm2_ps));
    // vextractf128 xmm5,ymm2,1
    xmm5 = _mm256_extractf128_ps(ymm2_ps, 1);
    // vmovaps XMMWORD ptr[r8+160],xmm3
    _mm_store_ps(ptr_p + 40, xmm3_ps);
    // vmovaps XMMWORD ptr[r8+176],xmm5
    _mm_store_ps(ptr_p + 44, xmm5);

    // vzeroupper
    _mm256_zeroupper();
}

// uc2s48_AVX2 proc ptr_t:dword,pitch:dword,ptr_pf:dword
// ptr_t = rcx
// pitch = edx
// ptr_pf = r8
extern "C" void uc2s48_AVX2(const uint8_t* ptr_t, int pitch, float* ptr_pf_raw) {
    int16_t* ptr_pf = reinterpret_cast<int16_t*>(ptr_pf_raw);
    // sub rsp,32
    // .allocstack 32
    // vmovdqu XMMWORD ptr[rsp],xmm6
    // .savexmm128 xmm6,0
    // vmovdqu XMMWORD ptr[rsp+16],xmm7
    // .savexmm128 xmm7,16
    // .endprolog

    // mov rax,rcx
    const uint8_t* rax = ptr_t;
    // movsxd rcx,edx
    int64_t rcx = pitch;
    // lea rdx,[rax+rcx*4]
    const uint8_t* rdx = rax + rcx * 4;

    // vmovq xmm0,QWORD PTR[rax]
    __m128i xmm0 = _mm_loadl_epi64((__m128i*)rax);
    // vmovd xmm1,dword ptr[rax+8]
    int32_t value;
    std::memcpy(&value, rax + 8, sizeof(value));
    __m128i xmm1 = _mm_cvtsi32_si128(value);
    // vmovd xmm2,dword ptr[rax+rcx*2]
    std::memcpy(&value, rax + rcx * 2, sizeof(value));
    __m128i xmm2 = _mm_cvtsi32_si128(value);
    // vmovq xmm3,QWORD PTR[rax+rcx*2+4]
    __m128i xmm3 = _mm_loadl_epi64((__m128i*)(rax + rcx * 2 + 4));
    // vmovq xmm4,QWORD PTR[rdx]
    __m128i xmm4 = _mm_loadl_epi64((__m128i*)rdx);
    // vmovd xmm5,dword ptr[rdx+8]
    std::memcpy(&value, rdx + 8, sizeof(value));
    __m128i xmm5 = _mm_cvtsi32_si128(value);
    // vmovd xmm6,dword ptr[rdx+rcx*2]
    std::memcpy(&value, rdx + rcx * 2, sizeof(value));
    __m128i xmm6 = _mm_cvtsi32_si128(value);
    // vmovq xmm7,QWORD PTR[rdx+rcx*2+4]
    __m128i xmm7 = _mm_loadl_epi64((__m128i*)(rdx + rcx * 2 + 4));

    // vpunpckldq xmm1,xmm1,xmm2
    xmm1 = _mm_unpacklo_epi32(xmm1, xmm2);
    // vpxor xmm2,xmm2,xmm2
    xmm2 = _mm_setzero_si128();
    // vpunpckldq xmm5,xmm5,xmm6
    xmm5 = _mm_unpacklo_epi32(xmm5, xmm6);

    // vpunpcklbw xmm0,xmm0,xmm2
    xmm0 = _mm_unpacklo_epi8(xmm0, xmm2);
    // vpunpcklbw xmm3,xmm3,xmm2
    xmm3 = _mm_unpacklo_epi8(xmm3, xmm2);
    // vpunpcklbw xmm1,xmm1,xmm2
    xmm1 = _mm_unpacklo_epi8(xmm1, xmm2);
    // vpunpcklbw xmm4,xmm4,xmm2
    xmm4 = _mm_unpacklo_epi8(xmm4, xmm2);
    // vpunpcklbw xmm5,xmm5,xmm2
    xmm5 = _mm_unpacklo_epi8(xmm5, xmm2);
    // vpunpcklbw xmm7,xmm7,xmm2
    xmm7 = _mm_unpacklo_epi8(xmm7, xmm2);

    // vmovdqa XMMWORD ptr[r8],xmm0
    _mm_store_si128((__m128i*)ptr_pf, xmm0);
    // vmovdqa XMMWORD ptr[r8+16],xmm1
    _mm_store_si128((__m128i*)(ptr_pf + 8), xmm1);
    // vmovdqa XMMWORD ptr[r8+32],xmm3
    _mm_store_si128((__m128i*)(ptr_pf + 16), xmm3);
    // vmovdqa XMMWORD ptr[r8+48],xmm4
    _mm_store_si128((__m128i*)(ptr_pf + 24), xmm4);
    // vmovdqa XMMWORD ptr[r8+64],xmm5
    _mm_store_si128((__m128i*)(ptr_pf + 32), xmm5);
    // vmovdqa XMMWORD ptr[r8+80],xmm7
    _mm_store_si128((__m128i*)(ptr_pf + 40), xmm7);

    // vzeroupper
    _mm256_zeroupper();
}

// uc2s64_AVX2 proc ptr_t:dword,pitch:dword,ptr_p:dword
// ptr_t = rcx
// pitch = edx
// ptr_p = r8
extern "C" void uc2s64_AVX2(const uint8_t* ptr_t, int pitch, float* ptr_p_raw) {
    int16_t* ptr_p = reinterpret_cast<int16_t*>(ptr_p_raw);
    // .endprolog

    // mov rax,rcx
    const uint8_t* rax = ptr_t;
    // movsxd rcx,edx
    int64_t rcx = pitch;
    // lea rdx,[rax+rcx*4]
    const uint8_t* rdx = rax + rcx * 4;
    // vpxor ymm4,ymm4,ymm4
    __m256i ymm4 = _mm256_setzero_si256();

    // test rax,15
    __m128i xmm0, xmm1, xmm2, xmm3;
    if ((reinterpret_cast<uintptr_t>(rax) & 15) == 0) {
        // vmovdqa xmm0,XMMWORD ptr[rax]
        xmm0 = _mm_load_si128((__m128i*)rax);
        // vmovdqa xmm1,XMMWORD PTR[rax+rcx*2]
        xmm1 = _mm_load_si128((__m128i*)(rax + rcx * 2));
        // vmovdqa xmm2,XMMWORD ptr[rdx]
        xmm2 = _mm_load_si128((__m128i*)rdx);
        // vmovdqa xmm3,XMMWORD PTR[rdx+rcx*2]
        xmm3 = _mm_load_si128((__m128i*)(rdx + rcx * 2));
    } else {
        // unaligned_3:
        // vmovdqu xmm0,XMMWORD ptr[rax]
        xmm0 = _mm_loadu_si128((__m128i*)rax);
        // vmovdqu xmm1,XMMWORD PTR[rax+rcx*2]
        xmm1 = _mm_loadu_si128((__m128i*)(rax + rcx * 2));
        // vmovdqu xmm2,XMMWORD ptr[rdx]
        xmm2 = _mm_loadu_si128((__m128i*)rdx);
        // vmovdqu xmm3,XMMWORD PTR[rdx+rcx*2]
        xmm3 = _mm_loadu_si128((__m128i*)(rdx + rcx * 2));
    }

    // suite_3:
    // vmovhlps xmm5,xmm4,xmm0
    __m128i xmm5 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm0)));
    // vinserti128 ymm0,ymm0,xmm5,1
    __m256i ymm0 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm0), xmm5, 1);
    // vmovhlps xmm5,xmm4,xmm1
    xmm5 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm1)));
    // vinserti128 ymm1,ymm1,xmm5,1
    __m256i ymm1 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm1), xmm5, 1);
    // vmovhlps xmm5,xmm4,xmm2
    xmm5 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm2)));
    // vinserti128 ymm2,ymm2,xmm5,1
    __m256i ymm2 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm2), xmm5, 1);
    // vmovhlps xmm5,xmm4,xmm3
    xmm5 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm4)), _mm_castsi128_ps(xmm3)));
    // vinserti128 ymm3,ymm3,xmm5,1
    __m256i ymm3 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm3), xmm5, 1);

    // vpunpcklbw ymm0,ymm0,ymm4
    ymm0 = _mm256_unpacklo_epi8(ymm0, ymm4);
    // vpunpcklbw ymm1,ymm1,ymm4
    ymm1 = _mm256_unpacklo_epi8(ymm1, ymm4);
    // vpunpcklbw ymm2,ymm2,ymm4
    ymm2 = _mm256_unpacklo_epi8(ymm2, ymm4);
    // vpunpcklbw ymm3,ymm3,ymm4
    ymm3 = _mm256_unpacklo_epi8(ymm3, ymm4);

    // vmovdqa YMMWORD ptr [r8],ymm0
    _mm256_store_si256((__m256i*)ptr_p, ymm0);
    // vmovdqa YMMWORD ptr [r8+32],ymm1
    _mm256_store_si256((__m256i*)(ptr_p + 16), ymm1);
    // vmovdqa YMMWORD ptr [r8+64],ymm2
    _mm256_store_si256((__m256i*)(ptr_p + 32), ymm2);
    // vmovdqa YMMWORD ptr [r8+96],ymm3
    _mm256_store_si256((__m256i*)(ptr_p + 48), ymm3);

    // vzeroupper
    _mm256_zeroupper();
}

extern "C" void dotProd_m32_m16_FMA3(
    const float* data_raw,    // rcx
    const float* weights_raw, // rdx
    float* vals_raw,          // r8
    int n,          // r9d
    int len,        // [rbp+48]
    const float* istd     // [rbp+56]
) {
    // レジスタの初期化
    const char* rdi = reinterpret_cast<const char*>(weights_raw);
    char* rax = reinterpret_cast<char*>(vals_raw);
    int rbx = n;
    int rsi = len;
    const char* r15 = reinterpret_cast<const char*>(data_raw);

    // 定数の設定
    const int r10 = 4;
    const int r11 = 16;
    const int r12 = 32;
    const int r13 = 128;
    const int r14 = 512;

    // nloop_2
    while (rbx != 0) {
        const char* rcx = r15;
        __m256 ymm0 = _mm256_setzero_ps();
        __m256 ymm1 = _mm256_setzero_ps();
        __m256 ymm2 = _mm256_setzero_ps();
        __m256 ymm3 = _mm256_setzero_ps();
        int rdx = rsi;

        // lloop_2
        while (rdx != 0) {
            // vmovaps ymm7,YMMWORD ptr[rcx]
            __m256 ymm7 = _mm256_load_ps((float*)rcx);
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)rdi), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+r12]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + r12)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+2*r12]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 2*r12)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+96]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 96)), ymm3);

            // vmovaps ymm7,YMMWORD ptr[rcx+r12]
            ymm7 = _mm256_load_ps((float*)(rcx + r12));
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi+r13]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + r13)), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+160]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 160)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+192]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 192)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+224]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 224)), ymm3);

            // vmovaps ymm7,YMMWORD ptr[rcx+2*r12]
            ymm7 = _mm256_load_ps((float*)(rcx + 2*r12));
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi+2*r13]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 2*r13)), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+288]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 288)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+320]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 320)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+352]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 352)), ymm3);

            // vmovaps ymm7,YMMWORD ptr[rcx+96]
            ymm7 = _mm256_load_ps((float*)(rcx + 96));
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi+384]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 384)), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+416]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 416)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+448]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 448)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+480]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 480)), ymm3);

            rcx += r13;
            rdi += r14;
            rdx -= r12;
        }

        // vextractf128 xmm4,ymm0,1
        __m128 xmm4 = _mm256_extractf128_ps(ymm0, 1);
        // vextractf128 xmm5,ymm1,1
        __m128 xmm5 = _mm256_extractf128_ps(ymm1, 1);
        // vextractf128 xmm6,ymm2,1
        __m128 xmm6 = _mm256_extractf128_ps(ymm2, 1);
        // vextractf128 xmm7,ymm3,1
        __m128 xmm7 = _mm256_extractf128_ps(ymm3, 1);

        // vaddps xmm0,xmm0,xmm4
        __m128 xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm4);
        // vaddps xmm1,xmm1,xmm5
        __m128 xmm1 = _mm_add_ps(_mm256_castps256_ps128(ymm1), xmm5);
        // vaddps xmm2,xmm2,xmm6
        __m128 xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm6);
        // vaddps xmm3,xmm3,xmm7
        __m128 xmm3 = _mm_add_ps(_mm256_castps256_ps128(ymm3), xmm7);

        // haddps xmm0,xmm1
        xmm0 = _mm_hadd_ps(xmm0, xmm1);
        // haddps xmm2,xmm3
        xmm2 = _mm_hadd_ps(xmm2, xmm3);
        // haddps xmm0,xmm2
        xmm0 = _mm_hadd_ps(xmm0, xmm2);

        // vmovaps XMMWORD ptr[rax],xmm0
        _mm_store_ps((float*)rax, xmm0);
        rax += r11;
        rbx -= r10;
    }

    // 最終処理
    const char* rcx = reinterpret_cast<const char*>(istd);
    rax = reinterpret_cast<char*>(vals_raw);
    // vmovss xmm7,dword ptr[rcx]
    __m128 xmm7 = _mm_load_ss((float*)rcx);
    int rdx = n;
    // vshufps xmm7,xmm7,xmm7,0
    xmm7 = _mm_shuffle_ps(xmm7, xmm7, 0);
    const __m256 ymm7_full = _mm256_broadcastss_ps(xmm7);
    int rcx2 = 0;
    // aloop_2
    while (rdx != 0) {
        // FMA化: vfmadd213ps ymm0,ymm7,YMMWORD ptr[rdi+rcx*4]
        __m256 ymm0 = _mm256_fmadd_ps(_mm256_load_ps((float*)(rax + rcx2*4)), ymm7_full,
            _mm256_load_ps((float*)(rdi + rcx2*4)));
        // FMA化: vfmadd213ps ymm2,ymm7,YMMWORD ptr[rdi+rcx*4+32]
        __m256 ymm2 = _mm256_fmadd_ps(_mm256_load_ps((float*)(rax + rcx2*4 + 32)), ymm7_full,
            _mm256_load_ps((float*)(rdi + rcx2*4 + 32)));
        // vmovaps YMMWORD ptr[rax+rcx*4],ymm0
        _mm256_store_ps((float*)(rax + rcx2*4), ymm0);
        // vmovaps YMMWORD ptr[rax+rcx*4+32],ymm2
        _mm256_store_ps((float*)(rax + rcx2*4 + 32), ymm2);
        rcx2 += r11;
        rdx -= r11;
    }

    _mm256_zeroupper();
}

// 元のアセンブラ関数の引数:
// dotProd_m48_m16_FMA3 proc data_:dword,weights:dword,vals:dword,n:dword,len:dword,istd:dword
// data_ = rcx
// weights = rdx
// vals = r8
// n = r9d
// len = [rbp+48]
// istd = [rbp+56]

extern "C" void dotProd_m48_m16_FMA3(
    const float* data_raw,    // rcx
    const float* weights_raw, // rdx
    float* vals_raw,          // r8
    int n,          // r9d
    int len,        // [rbp+48]
    const float* istd     // [rbp+56]
) {
    // レジスタの初期化
    const char* rdi = reinterpret_cast<const char*>(weights_raw);
    char* rax = reinterpret_cast<char*>(vals_raw);
    int rbx = n;
    int rsi = len;
    const char* r15 = reinterpret_cast<const char*>(data_raw);

    // 定数の設定
    const int r10 = 4;
    const int r11 = 16;
    const int r12 = 48;
    const int r13 = 192;
    const int r14 = 768;

    // nloop2_2
    while (rbx != 0) {
        const char* rcx = r15;
        __m256 ymm0 = _mm256_setzero_ps();
        __m256 ymm1 = _mm256_setzero_ps();
        __m256 ymm2 = _mm256_setzero_ps();
        __m256 ymm3 = _mm256_setzero_ps();
        int rdx = rsi;

        // lloop2_2
        while (rdx != 0) {
            // vmovaps ymm7,YMMWORD ptr[rcx]
            __m256 ymm7 = _mm256_load_ps((float*)rcx);
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)rdi), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+2*r11]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 2*r11)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+4*r11]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 4*r11)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+96]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 96)), ymm3);

            // vmovaps ymm7,YMMWORD ptr[rcx+2*r11]
            ymm7 = _mm256_load_ps((float*)(rcx + 2*r11));
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi+8*r11]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 8*r11)), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+160]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 160)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+192]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 192)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+224]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 224)), ymm3);

            // vmovaps ymm7,YMMWORD ptr[rcx+4*r11]
            ymm7 = _mm256_load_ps((float*)(rcx + 4*r11));
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi+256]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 256)), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+288]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 288)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+320]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 320)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+352]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 352)), ymm3);

            // vmovaps ymm7,YMMWORD ptr[rcx+96]
            ymm7 = _mm256_load_ps((float*)(rcx + 96));
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi+2*r13]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 2*r13)), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+416]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 416)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+448]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 448)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+480]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 480)), ymm3);

            // vmovaps ymm7,YMMWORD ptr[rcx+8*r11]
            ymm7 = _mm256_load_ps((float*)(rcx + 8*r11));
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi+512]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 512)), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+544]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 544)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+576]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 576)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+608]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 608)), ymm3);

            // vmovaps ymm7,YMMWORD ptr[rcx+160]
            ymm7 = _mm256_load_ps((float*)(rcx + 160));
            // vfmadd231ps ymm0,ymm7,YMMWORD ptr[rdi+640]
            ymm0 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 640)), ymm0);
            // vfmadd231ps ymm1,ymm7,YMMWORD ptr[rdi+672]
            ymm1 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 672)), ymm1);
            // vfmadd231ps ymm2,ymm7,YMMWORD ptr[rdi+704]
            ymm2 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 704)), ymm2);
            // vfmadd231ps ymm3,ymm7,YMMWORD ptr[rdi+736]
            ymm3 = _mm256_fmadd_ps(ymm7, _mm256_load_ps((float*)(rdi + 736)), ymm3);

            rcx += r13;
            rdi += r14;
            rdx -= r12;
        }

        // vextractf128 xmm4,ymm0,1
        __m128 xmm4 = _mm256_extractf128_ps(ymm0, 1);
        // vextractf128 xmm5,ymm1,1
        __m128 xmm5 = _mm256_extractf128_ps(ymm1, 1);
        // vextractf128 xmm6,ymm2,1
        __m128 xmm6 = _mm256_extractf128_ps(ymm2, 1);
        // vextractf128 xmm7,ymm3,1
        __m128 xmm7 = _mm256_extractf128_ps(ymm3, 1);

        // vaddps xmm0,xmm0,xmm4
        __m128 xmm0 = _mm_add_ps(_mm256_castps256_ps128(ymm0), xmm4);
        // vaddps xmm1,xmm1,xmm5
        __m128 xmm1 = _mm_add_ps(_mm256_castps256_ps128(ymm1), xmm5);
        // vaddps xmm2,xmm2,xmm6
        __m128 xmm2 = _mm_add_ps(_mm256_castps256_ps128(ymm2), xmm6);
        // vaddps xmm3,xmm3,xmm7
        __m128 xmm3 = _mm_add_ps(_mm256_castps256_ps128(ymm3), xmm7);

        // haddps xmm0,xmm1
        xmm0 = _mm_hadd_ps(xmm0, xmm1);
        // haddps xmm2,xmm3
        xmm2 = _mm_hadd_ps(xmm2, xmm3);
        // haddps xmm0,xmm2
        xmm0 = _mm_hadd_ps(xmm0, xmm2);

        // vmovaps XMMWORD ptr[rax],xmm0
        _mm_store_ps((float*)rax, xmm0);
        rax += r11;
        rbx -= r10;
    }

    // 最終処理
    const char* rcx = reinterpret_cast<const char*>(istd);
    rax = reinterpret_cast<char*>(vals_raw);
    // vmovss xmm7,dword ptr[rcx]
    __m128 xmm7 = _mm_load_ss((float*)rcx);
    int rdx = n;
    // vshufps xmm7,xmm7,xmm7,0
    xmm7 = _mm_shuffle_ps(xmm7, xmm7, 0);
    int rcx2 = 0;
    // vinsertf128 ymm7,ymm7,xmm7,1
    const __m256 ymm7_full = _mm256_broadcastss_ps(xmm7);

    // aloop2_2
    while (rdx != 0) {
        // FMA化: vfmadd213ps ymm0,ymm7,YMMWORD ptr[rdi+rcx*4]
        __m256 ymm0 = _mm256_fmadd_ps(_mm256_load_ps((float*)(rax + rcx2*4)), ymm7_full,
            _mm256_load_ps((float*)(rdi + rcx2*4)));
        // FMA化: vfmadd213ps ymm2,ymm7,YMMWORD ptr[rdi+rcx*4+32]
        __m256 ymm2 = _mm256_fmadd_ps(_mm256_load_ps((float*)(rax + rcx2*4 + 32)), ymm7_full,
            _mm256_load_ps((float*)(rdi + rcx2*4 + 32)));
        // vmovaps YMMWORD ptr[rax+rcx*4],ymm0
        _mm256_store_ps((float*)(rax + rcx2*4), ymm0);
        // vmovaps YMMWORD ptr[rax+rcx*4+32],ymm2
        _mm256_store_ps((float*)(rax + rcx2*4 + 32), ymm2);
        rcx2 += r11;
        rdx -= r11;
    }

    _mm256_zeroupper();
}

// 元のアセンブラ関数の引数:
// dotProd_m32_m16_i16_AVX2 proc dataf:dword,weightsf:dword,vals:dword,n:dword,len:dword,istd:dword
// dataf = rcx
// weightsf = rdx
// vals = r8
// n = r9d
// len = [rbp+48]
// istd = [rbp+56]

extern "C" void dotProd_m32_m16_i16_AVX2(
    const float* data_raw,    // rcx
    const float* weights_raw, // rdx
    float* vals_raw,          // r8
    int n,          // r9d
    int len,        // [rbp+48]
    const float* istd     // [rbp+56]
) {
    // レジスタの初期化
    const char* rdi = reinterpret_cast<const char*>(weights_raw);
    char* rax = reinterpret_cast<char*>(vals_raw);
    int rbx = n;
    int rsi = len;
    const char* r15 = reinterpret_cast<const char*>(data_raw);

    // 定数の設定
    const int r10 = 4;
    const int r11 = 16;
    const int r12 = 32;
    const int r13 = 64;
    const int r14 = 256;

    // nloop_3
    while (rbx != 0) {
        const char* rcx = r15;
        __m256i ymm0 = _mm256_setzero_si256();
        __m256i ymm1 = _mm256_setzero_si256();
        __m256i ymm2 = _mm256_setzero_si256();
        __m256i ymm3 = _mm256_setzero_si256();
        int rdx = rsi;

        // lloop_3
        while (rdx != 0) {
            // vmovdqa ymm7,YMMWORD ptr [rcx]
            __m256i ymm7 = _mm256_load_si256((__m256i*)rcx);
            // vpmaddwd ymm4,ymm7,YMMWORD ptr [rdi]
            __m256i ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)rdi));
            // vpmaddwd ymm5,ymm7,YMMWORD ptr [rdi+r12]
            __m256i ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + r12)));
            // vpmaddwd ymm6,ymm7,YMMWORD ptr [rdi+r13]
            __m256i ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + r13)));
            // vpmaddwd ymm7,ymm7,YMMWORD ptr [rdi+96]
            ymm7 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 96)));
            // vpaddd ymm0,ymm0,ymm4
            ymm0 = _mm256_add_epi32(ymm0, ymm4);
            // vpaddd ymm1,ymm1,ymm5
            ymm1 = _mm256_add_epi32(ymm1, ymm5);
            // vpaddd ymm2,ymm2,ymm6
            ymm2 = _mm256_add_epi32(ymm2, ymm6);
            // vpaddd ymm3,ymm3,ymm7
            ymm3 = _mm256_add_epi32(ymm3, ymm7);

            // vmovdqa ymm7,YMMWORD ptr [rcx+r12]
            ymm7 = _mm256_load_si256((__m256i*)(rcx + r12));
            // vpmaddwd ymm4,ymm7,YMMWORD ptr [rdi+r13*2]
            ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + r13*2)));
            // vpmaddwd ymm5,ymm7,YMMWORD ptr [rdi+160]
            ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 160)));
            // vpmaddwd ymm6,ymm7,YMMWORD ptr [rdi+192]
            ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 192)));
            // vpmaddwd ymm7,ymm7,YMMWORD ptr [rdi+224]
            ymm7 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 224)));
            // vpaddd ymm0,ymm0,ymm4
            ymm0 = _mm256_add_epi32(ymm0, ymm4);
            // vpaddd ymm1,ymm1,ymm5
            ymm1 = _mm256_add_epi32(ymm1, ymm5);
            // vpaddd ymm2,ymm2,ymm6
            ymm2 = _mm256_add_epi32(ymm2, ymm6);
            // vpaddd ymm3,ymm3,ymm7
            ymm3 = _mm256_add_epi32(ymm3, ymm7);

            rcx += r13;
            rdi += r14;
            rdx -= r12;
        }

        // vextracti128 xmm4,ymm0,1
        __m128i xmm4 = _mm256_extracti128_si256(ymm0, 1);
        // vextracti128 xmm5,ymm1,1
        __m128i xmm5 = _mm256_extracti128_si256(ymm1, 1);
        // vextracti128 xmm6,ymm2,1
        __m128i xmm6 = _mm256_extracti128_si256(ymm2, 1);
        // vextracti128 xmm7,ymm3,1
        __m128i xmm7 = _mm256_extracti128_si256(ymm3, 1);

        // vpaddd xmm0,xmm0,xmm4
        __m128i xmm0 = _mm_add_epi32(_mm256_castsi256_si128(ymm0), xmm4);
        // vpaddd xmm1,xmm1,xmm5
        __m128i xmm1 = _mm_add_epi32(_mm256_castsi256_si128(ymm1), xmm5);
        // vpaddd xmm2,xmm2,xmm6
        __m128i xmm2 = _mm_add_epi32(_mm256_castsi256_si128(ymm2), xmm6);
        // vpaddd xmm3,xmm3,xmm7
        __m128i xmm3 = _mm_add_epi32(_mm256_castsi256_si128(ymm3), xmm7);

        // vpunpckhqdq xmm4,xmm0,xmm1
        xmm4 = _mm_unpackhi_epi64(xmm0, xmm1);
        // vpunpckhqdq xmm5,xmm2,xmm3
        xmm5 = _mm_unpackhi_epi64(xmm2, xmm3);
        // vpunpcklqdq xmm0,xmm0,xmm1
        xmm0 = _mm_unpacklo_epi64(xmm0, xmm1);
        // vpunpcklqdq xmm2,xmm2,xmm3
        xmm2 = _mm_unpacklo_epi64(xmm2, xmm3);

        // vpaddd xmm0,xmm0,xmm4
        xmm0 = _mm_add_epi32(xmm0, xmm4);
        // vpaddd xmm2,xmm2,xmm5
        xmm2 = _mm_add_epi32(xmm2, xmm5);

        // vshufps xmm6,xmm0,xmm2,221
        __m128 xmm6_ps = _mm_shuffle_ps(_mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2), 221);
        // vshufps xmm0,xmm0,xmm2,136
        __m128 xmm0_ps = _mm_shuffle_ps(_mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2), 136);

        // vpaddd xmm6,xmm6,xmm0
        __m128i xmm6_final = _mm_add_epi32(_mm_castps_si128(xmm6_ps), _mm_castps_si128(xmm0_ps));

        // vmovdqa XMMWORD ptr [rax],xmm6
        _mm_store_si128((__m128i*)rax, xmm6_final);
        rax += r11;
        rbx -= r10;
    }

    // 最終処理
    const char* rcx = reinterpret_cast<const char*>(istd);
    rax = reinterpret_cast<char*>(vals_raw);
    // vmovss xmm7,dword ptr[rcx]
    __m128 xmm7 = _mm_load_ss((float*)rcx);
    int rdx = n;
    // vpshufd xmm7,xmm7,0
    xmm7 = _mm_shuffle_ps(xmm7, xmm7, 0);
    int rcx2 = 0;

    // aloop_3
    while (rdx != 0) {
        // vmovdqa ymm0,YMMWORD ptr[rax+rcx*4]
        __m256i ymm0 = _mm256_load_si256((__m256i*)(rax + rcx2*4));
        // vmovdqa ymm2,YMMWORD ptr[rax+rcx*4+32]
        __m256i ymm2 = _mm256_load_si256((__m256i*)(rax + rcx2*4 + 32));
        // vcvtdq2ps ymm0,ymm0
        __m256 ymm0_ps = _mm256_cvtepi32_ps(ymm0);
        // vcvtdq2ps ymm2,ymm2
        __m256 ymm2_ps = _mm256_cvtepi32_ps(ymm2);
        // vextractf128 xmm1,ymm0,1
        __m128 xmm1 = _mm256_extractf128_ps(ymm0_ps, 1);
        // vextractf128 xmm3,ymm2,1
        __m128 xmm3 = _mm256_extractf128_ps(ymm2_ps, 1);

        // vmulps xmm0,xmm0,XMMWORD ptr[rdi+rcx*8]
        __m128 xmm0 = _mm_mul_ps(_mm256_castps256_ps128(ymm0_ps), _mm_load_ps((float*)(rdi + rcx2*8)));
        // vmulps xmm1,xmm1,XMMWORD ptr[rdi+rcx*8+32]
        xmm1 = _mm_mul_ps(xmm1, _mm_load_ps((float*)(rdi + rcx2*8 + 32)));
        // vmulps xmm2,xmm2,XMMWORD ptr[rdi+rcx*8+64]
        __m128 xmm2 = _mm_mul_ps(_mm256_castps256_ps128(ymm2_ps), _mm_load_ps((float*)(rdi + rcx2*8 + 64)));
        // vmulps xmm3,xmm3,XMMWORD ptr[rdi+rcx*8+96]
        xmm3 = _mm_mul_ps(xmm3, _mm_load_ps((float*)(rdi + rcx2*8 + 96)));

        // FMA化: vfmadd213ps xmm0,xmm7,XMMWORD ptr[rdi+rcx*8+16]
        xmm0 = _mm_fmadd_ps(xmm0, xmm7, _mm_load_ps((float*)(rdi + rcx2*8 + 16)));
        // FMA化: vfmadd213ps xmm1,xmm7,XMMWORD ptr[rdi+rcx*8+48]
        xmm1 = _mm_fmadd_ps(xmm1, xmm7, _mm_load_ps((float*)(rdi + rcx2*8 + 48)));
        // FMA化: vfmadd213ps xmm2,xmm7,XMMWORD ptr[rdi+rcx*8+80]
        xmm2 = _mm_fmadd_ps(xmm2, xmm7, _mm_load_ps((float*)(rdi + rcx2*8 + 80)));
        // FMA化: vfmadd213ps xmm3,xmm7,XMMWORD ptr[rdi+rcx*8+112]
        xmm3 = _mm_fmadd_ps(xmm3, xmm7, _mm_load_ps((float*)(rdi + rcx2*8 + 112)));

        // vmovaps XMMWORD ptr[rax+rcx*4],xmm0
        _mm_store_ps((float*)(rax + rcx2*4), xmm0);
        // vmovaps XMMWORD ptr[rax+rcx*4+16],xmm1
        _mm_store_ps((float*)(rax + rcx2*4 + 16), xmm1);
        // vmovaps XMMWORD ptr[rax+rcx*4+32],xmm2
        _mm_store_ps((float*)(rax + rcx2*4 + 32), xmm2);
        // vmovaps XMMWORD ptr[rax+rcx*4+48],xmm3
        _mm_store_ps((float*)(rax + rcx2*4 + 48), xmm3);

        rcx2 += r11;
        rdx -= r11;
    }

    _mm256_zeroupper();
}

// 元のアセンブラ関数の引数:
// dotProd_m48_m16_i16_AVX2 proc dataf:dword,weightsf:dword,vals:dword,n:dword,len:dword,istd:dword
// dataf = rcx
// weightsf = rdx
// vals = r8
// n = r9d
// len = [rbp+48]
// istd = [rbp+56]

extern "C" void dotProd_m48_m16_i16_AVX2(
    const float* data_raw,    // rcx
    const float* weights_raw, // rdx
    float* vals_raw,          // r8
    int n,          // r9d
    int len,        // [rbp+48]
    const float* istd     // [rbp+56]
) {
    // レジスタの初期化
    const char* rdi = reinterpret_cast<const char*>(weights_raw);
    char* rax = reinterpret_cast<char*>(vals_raw);
    int rbx = n;
    int rsi = len;
    const char* r15 = reinterpret_cast<const char*>(data_raw);

    // 定数の設定
    const int r10 = 4;
    const int r11 = 16;
    const int r12 = 48;
    const int r13 = 96;
    const int r14 = 384;

    // nloop_4
    while (rbx != 0) {
        const char* rcx = r15;
        __m256i ymm0 = _mm256_setzero_si256();
        __m256i ymm1 = _mm256_setzero_si256();
        __m256i ymm2 = _mm256_setzero_si256();
        __m256i ymm3 = _mm256_setzero_si256();
        int rdx = rsi;

        // lloop_4
        while (rdx != 0) {
            // vmovdqa ymm7,YMMWORD ptr [rcx]
            __m256i ymm7 = _mm256_load_si256((__m256i*)rcx);
            // vpmaddwd ymm4,ymm7,YMMWORD ptr [rdi]
            __m256i ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)rdi));
            // vpmaddwd ymm5,ymm7,YMMWORD ptr [rdi+r11*2]
            __m256i ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + r11*2)));
            // vpmaddwd ymm6,ymm7,YMMWORD ptr [rdi+r11*4]
            __m256i ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + r11*4)));
            // vpmaddwd ymm7,ymm7,YMMWORD ptr [rdi+r13]
            ymm7 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + r13)));
            // vpaddd ymm0,ymm0,ymm4
            ymm0 = _mm256_add_epi32(ymm0, ymm4);
            // vpaddd ymm1,ymm1,ymm5
            ymm1 = _mm256_add_epi32(ymm1, ymm5);
            // vpaddd ymm2,ymm2,ymm6
            ymm2 = _mm256_add_epi32(ymm2, ymm6);
            // vpaddd ymm3,ymm3,ymm7
            ymm3 = _mm256_add_epi32(ymm3, ymm7);

            // vmovdqa ymm7,YMMWORD ptr [rcx+r11*2]
            ymm7 = _mm256_load_si256((__m256i*)(rcx + r11*2));
            // vpmaddwd ymm4,ymm7,YMMWORD ptr [rdi+r11*8]
            ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + r11*8)));
            // vpmaddwd ymm5,ymm7,YMMWORD ptr [rdi+160]
            ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 160)));
            // vpmaddwd ymm6,ymm7,YMMWORD ptr [rdi+192]
            ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 192)));
            // vpmaddwd ymm7,ymm7,YMMWORD ptr [rdi+224]
            ymm7 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 224)));
            // vpaddd ymm0,ymm0,ymm4
            ymm0 = _mm256_add_epi32(ymm0, ymm4);
            // vpaddd ymm1,ymm1,ymm5
            ymm1 = _mm256_add_epi32(ymm1, ymm5);
            // vpaddd ymm2,ymm2,ymm6
            ymm2 = _mm256_add_epi32(ymm2, ymm6);
            // vpaddd ymm3,ymm3,ymm7
            ymm3 = _mm256_add_epi32(ymm3, ymm7);

            // vmovdqa ymm7,YMMWORD ptr [rcx+r11*4]
            ymm7 = _mm256_load_si256((__m256i*)(rcx + r11*4));
            // vpmaddwd ymm4,ymm7,YMMWORD ptr [rdi+256]
            ymm4 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 256)));
            // vpmaddwd ymm5,ymm7,YMMWORD ptr [rdi+288]
            ymm5 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 288)));
            // vpmaddwd ymm6,ymm7,YMMWORD ptr [rdi+320]
            ymm6 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 320)));
            // vpmaddwd ymm7,ymm7,YMMWORD ptr [rdi+352]
            ymm7 = _mm256_madd_epi16(ymm7, _mm256_load_si256((__m256i*)(rdi + 352)));
            // vpaddd ymm0,ymm0,ymm4
            ymm0 = _mm256_add_epi32(ymm0, ymm4);
            // vpaddd ymm1,ymm1,ymm5
            ymm1 = _mm256_add_epi32(ymm1, ymm5);
            // vpaddd ymm2,ymm2,ymm6
            ymm2 = _mm256_add_epi32(ymm2, ymm6);
            // vpaddd ymm3,ymm3,ymm7
            ymm3 = _mm256_add_epi32(ymm3, ymm7);

            rcx += r13;
            rdi += r14;
            rdx -= r12;
        }

        // vextracti128 xmm4,ymm0,1
        __m128i xmm4 = _mm256_extracti128_si256(ymm0, 1);
        // vextracti128 xmm5,ymm1,1
        __m128i xmm5 = _mm256_extracti128_si256(ymm1, 1);
        // vextracti128 xmm6,ymm2,1
        __m128i xmm6 = _mm256_extracti128_si256(ymm2, 1);
        // vextracti128 xmm7,ymm3,1
        __m128i xmm7 = _mm256_extracti128_si256(ymm3, 1);

        // vpaddd xmm0,xmm0,xmm4
        __m128i xmm0 = _mm_add_epi32(_mm256_castsi256_si128(ymm0), xmm4);
        // vpaddd xmm1,xmm1,xmm5
        __m128i xmm1 = _mm_add_epi32(_mm256_castsi256_si128(ymm1), xmm5);
        // vpaddd xmm2,xmm2,xmm6
        __m128i xmm2 = _mm_add_epi32(_mm256_castsi256_si128(ymm2), xmm6);
        // vpaddd xmm3,xmm3,xmm7
        __m128i xmm3 = _mm_add_epi32(_mm256_castsi256_si128(ymm3), xmm7);

        // vpunpckhqdq xmm4,xmm0,xmm1
        xmm4 = _mm_unpackhi_epi64(xmm0, xmm1);
        // vpunpckhqdq xmm5,xmm2,xmm3
        xmm5 = _mm_unpackhi_epi64(xmm2, xmm3);
        // vpunpcklqdq xmm0,xmm0,xmm1
        xmm0 = _mm_unpacklo_epi64(xmm0, xmm1);
        // vpunpcklqdq xmm2,xmm2,xmm3
        xmm2 = _mm_unpacklo_epi64(xmm2, xmm3);

        // vpaddd xmm0,xmm0,xmm4
        xmm0 = _mm_add_epi32(xmm0, xmm4);
        // vpaddd xmm2,xmm2,xmm5
        xmm2 = _mm_add_epi32(xmm2, xmm5);

        // vshufps xmm6,xmm0,xmm2,221
        __m128 xmm6_ps = _mm_shuffle_ps(_mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2), 221);
        // vshufps xmm0,xmm0,xmm2,136
        __m128 xmm0_ps = _mm_shuffle_ps(_mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2), 136);

        // vpaddd xmm6,xmm6,xmm0
        __m128i xmm6_final = _mm_add_epi32(_mm_castps_si128(xmm6_ps), _mm_castps_si128(xmm0_ps));

        // vmovdqa XMMWORD ptr [rax],xmm6
        _mm_store_si128((__m128i*)rax, xmm6_final);
        rax += r11;
        rbx -= r10;
    }

    // 最終処理
    const char* rcx = reinterpret_cast<const char*>(istd);
    rax = reinterpret_cast<char*>(vals_raw);
    // vmovss xmm7,dword ptr[rcx]
    __m128 xmm7 = _mm_load_ss((float*)rcx);
    int rdx = n;
    // vpshufd xmm7,xmm7,0
    xmm7 = _mm_shuffle_ps(xmm7, xmm7, 0);
    int rcx2 = 0;
    // aloop_4
    while (rdx != 0) {
        // vmovdqa ymm0,YMMWORD ptr[rax+rcx*4]
        __m256i ymm0 = _mm256_load_si256((__m256i*)(rax + rcx2*4));
        // vmovdqa ymm2,YMMWORD ptr[rax+rcx*4+32]
        __m256i ymm2 = _mm256_load_si256((__m256i*)(rax + rcx2*4 + 32));
        // vcvtdq2ps ymm0,ymm0
        __m256 ymm0_ps = _mm256_cvtepi32_ps(ymm0);
        // vcvtdq2ps ymm2,ymm2
        __m256 ymm2_ps = _mm256_cvtepi32_ps(ymm2);
        // vextractf128 xmm1,ymm0,1
        __m128 xmm1 = _mm256_extractf128_ps(ymm0_ps, 1);
        // vextractf128 xmm3,ymm2,1
        __m128 xmm3 = _mm256_extractf128_ps(ymm2_ps, 1);

        // vmulps xmm0,xmm0,XMMWORD ptr[rdi+rcx*8]
        __m128 xmm0 = _mm_mul_ps(_mm256_castps256_ps128(ymm0_ps), _mm_load_ps((float*)(rdi + rcx2*8)));
        // vmulps xmm1,xmm1,XMMWORD ptr[rdi+rcx*8+32]
        xmm1 = _mm_mul_ps(xmm1, _mm_load_ps((float*)(rdi + rcx2*8 + 32)));
        // vmulps xmm2,xmm2,XMMWORD ptr[rdi+rcx*8+64]
        __m128 xmm2 = _mm_mul_ps(_mm256_castps256_ps128(ymm2_ps), _mm_load_ps((float*)(rdi + rcx2*8 + 64)));
        // vmulps xmm3,xmm3,XMMWORD ptr[rdi+rcx*8+96]
        xmm3 = _mm_mul_ps(xmm3, _mm_load_ps((float*)(rdi + rcx2*8 + 96)));

        // FMA化: vfmadd213ps xmm0,xmm7,XMMWORD ptr[rdi+rcx*8+16]
        xmm0 = _mm_fmadd_ps(xmm0, xmm7, _mm_load_ps((float*)(rdi + rcx2*8 + 16)));
        // FMA化: vfmadd213ps xmm1,xmm7,XMMWORD ptr[rdi+rcx*8+48]
        xmm1 = _mm_fmadd_ps(xmm1, xmm7, _mm_load_ps((float*)(rdi + rcx2*8 + 48)));
        // FMA化: vfmadd213ps xmm2,xmm7,XMMWORD ptr[rdi+rcx*8+80]
        xmm2 = _mm_fmadd_ps(xmm2, xmm7, _mm_load_ps((float*)(rdi + rcx2*8 + 80)));
        // FMA化: vfmadd213ps xmm3,xmm7,XMMWORD ptr[rdi+rcx*8+112]
        xmm3 = _mm_fmadd_ps(xmm3, xmm7, _mm_load_ps((float*)(rdi + rcx2*8 + 112)));

        // vmovaps XMMWORD ptr[rax+rcx*4],xmm0
        _mm_store_ps((float*)(rax + rcx2*4), xmm0);
        // vmovaps XMMWORD ptr[rax+rcx*4+16],xmm1
        _mm_store_ps((float*)(rax + rcx2*4 + 16), xmm1);
        // vmovaps XMMWORD ptr[rax+rcx*4+32],xmm2
        _mm_store_ps((float*)(rax + rcx2*4 + 32), xmm2);
        // vmovaps XMMWORD ptr[rax+rcx*4+48],xmm3
        _mm_store_ps((float*)(rax + rcx2*4 + 48), xmm3);

        rcx2 += r11;
        rdx -= r11;
    }

    _mm256_zeroupper();
}

// 元のアセンブラ関数の引数:
// e0_m16_FMA3 proc ptr_s:dword,n:dword
// ptr_s = rcx
// n = edx

extern "C" void e0_m16_FMA3(
    float* ptr_s,   // rcx
    int n           // edx
) {
    // レジスタの初期化
    char* rax = reinterpret_cast<char*>(ptr_s);
    int rcx = n;

    // 定数の設定
    const int rdx = 16;
    const int r8 = 32;
    const int r10 = 64;

    // 定数のロード
    // vmovdqa ymm2,YMMWORD ptr exp_hi
    __m256 ymm2 = _mm256_load_ps((float*)&exp_hi);
    // vmovdqa ymm3,YMMWORD ptr exp_lo
    __m256 ymm3 = _mm256_load_ps((float*)&exp_lo);
    // vmovdqa ymm4,YMMWORD ptr e0_mult
    __m256 ymm4 = _mm256_load_ps((float*)&e0_mult);
    // vmovdqa ymm5,YMMWORD ptr e0_bias
    __m256 ymm5 = _mm256_load_ps((float*)&e0_bias);

    // eloop16_2
    while (rcx != 0) {
        // vmovaps ymm0,YMMWORD ptr [rax]
        __m256 ymm0 = _mm256_load_ps((float*)rax);
        // vmovaps ymm1,YMMWORD ptr [rax+r8]
        __m256 ymm1 = _mm256_load_ps((float*)(rax + r8));
        // vminps ymm0,ymm0,ymm2
        ymm0 = _mm256_min_ps(ymm0, ymm2);
        // vminps ymm1,ymm1,ymm2
        ymm1 = _mm256_min_ps(ymm1, ymm2);
        // vmaxps ymm0,ymm0,ymm3
        ymm0 = _mm256_max_ps(ymm0, ymm3);
        // vmaxps ymm1,ymm1,ymm3
        ymm1 = _mm256_max_ps(ymm1, ymm3);

        // vfmadd213ps ymm0,ymm4,ymm5
        ymm0 = _mm256_fmadd_ps(ymm0, ymm4, ymm5);
        // vfmadd213ps ymm1,ymm4,ymm5
        ymm1 = _mm256_fmadd_ps(ymm1, ymm4, ymm5);

        // vcvtps2dq ymm0,ymm0
        __m256i ymm0_i = _mm256_cvtps_epi32(ymm0);
        // vcvtps2dq ymm1,ymm1
        __m256i ymm1_i = _mm256_cvtps_epi32(ymm1);

        // vmovaps YMMWORD ptr [rax],ymm0
        _mm256_store_ps((float*)rax, _mm256_castsi256_ps(ymm0_i));
        // vmovaps YMMWORD ptr [rax+r8],ymm1
        _mm256_store_ps((float*)(rax + r8), _mm256_castsi256_ps(ymm1_i));

        rax += r10;
        rcx -= rdx;
    }

    _mm256_zeroupper();
}

// 元のアセンブラ関数の引数:
// e1_m16_AVX2 proc ptr_s:dword,n:dword
// ptr_s = rcx
// n = edx

extern "C" void e1_m16_AVX2(
    float* ptr_s,   // rcx
    int n           // edx
) {
    // レジスタの初期化
    char* rax = reinterpret_cast<char*>(ptr_s);
    int rcx = n;

    // 定数の設定
    const int rdx = 8;
    const int r9 = 32;

    // 定数のロード
    // vmovdqa ymm3,YMMWORD ptr exp_hi
    __m256 ymm3 = _mm256_load_ps((float*)&exp_hi);
    // vmovdqa ymm4,YMMWORD ptr exp_lo
    __m256 ymm4 = _mm256_load_ps((float*)&exp_lo);
    // vmovdqa ymm5,YMMWORD ptr e1_scale
    __m256 ymm5 = _mm256_load_ps((float*)&e1_scale);
    // vmovdqa ymm6,YMMWORD ptr e1_bias
    __m256 ymm6 = _mm256_load_ps((float*)&e1_bias_ps);
    // vmovdqa ymm7,YMMWORD ptr e1_c1
    __m256 ymm7 = _mm256_load_ps((float*)&e1_c1);
    // vmovdqa ymm8,YMMWORD ptr e1_c2
    __m256 ymm8 = _mm256_load_ps((float*)&e1_c2);
    // vmovdqa ymm9,YMMWORD ptr e1_c0
    __m256 ymm9 = _mm256_load_ps((float*)&e1_c0);

    // eloop8
    while (rcx != 0) {
        // vmovaps ymm0,YMMWORD ptr [rax]
        __m256 ymm0 = _mm256_load_ps((float*)rax);
        // vminps ymm0,ymm0,ymm3
        ymm0 = _mm256_min_ps(ymm0, ymm3);
        // vmaxps ymm0,ymm0,ymm4
        ymm0 = _mm256_max_ps(ymm0, ymm4);
        // FMA化: vfmadd231ps ymm1,ymm0,ymm5
        __m256 ymm1 = _mm256_fmadd_ps(ymm0, ymm5, ymm6);
        // vpslld ymm2,ymm1,23
        __m256i ymm2 = _mm256_slli_epi32(_mm256_castps_si256(ymm1), 23);
        // vsubps ymm1,ymm1,ymm6
        ymm1 = _mm256_sub_ps(ymm1, ymm6);
        // FMA化: vfmsub213ps ymm0,ymm5,ymm1
        ymm0 = _mm256_fmsub_ps(ymm0, ymm5, ymm1);
        // FMA化: vfmadd213ps ymm1,ymm7,ymm9
        ymm1 = _mm256_fmadd_ps(ymm0, ymm7, ymm9);
        // vmulps ymm0,ymm0,ymm0
        ymm0 = _mm256_mul_ps(ymm0, ymm0);
        // FMA化: vfmadd231ps ymm1,ymm0,ymm8
        ymm0 = _mm256_fmadd_ps(ymm0, ymm8, ymm1);
        // vpaddd ymm0,ymm0,ymm2
        ymm0 = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(ymm0), ymm2));
        // vmovaps YMMWORD ptr [rax],ymm0
        _mm256_store_ps((float*)rax, ymm0);

        rax += r9;
        rcx -= rdx;
    }

    _mm256_zeroupper();
}

// 元のアセンブラ関数の引数:
// e2_m16_AVX2 proc ptr_s:dword,n:dword
// ptr_s = rcx
// n = edx

extern "C" void e2_m16_AVX2(
    float* ptr_s,   // rcx
    int n           // edx
) {
    // レジスタの初期化
    char* rax = reinterpret_cast<char*>(ptr_s);
    int rcx = n;

    // 定数の設定
    const int rdx = 8;
    const int r8 = 32;

    // 定数のロード
    // vmovdqa ymm7,YMMWORD ptr exp_hi
    __m256 ymm7 = _mm256_load_ps((float*)&exp_hi);
    // vmovdqa ymm8,YMMWORD ptr exp_lo
    __m256 ymm8 = _mm256_load_ps((float*)&exp_lo);
    // vmovdqa ymm9,YMMWORD ptr exp_rln2
    __m256 ymm9 = _mm256_load_ps((float*)&exp_rln2);
    // vmovdqa ymm10,YMMWORD ptr am_0p5
    __m256 ymm10 = _mm256_load_ps((float*)&am_0p5);
    // vmovdqa ymm11,YMMWORD ptr epi32_1
    __m256i ymm11 = _mm256_load_si256((__m256i*)&epi32_1);
    // vmovdqa ymm12,YMMWORD ptr exp_c2
    __m256 ymm12 = _mm256_load_ps((float*)&exp_c2);
    // vmovdqa ymm13,YMMWORD ptr exp_c1
    __m256 ymm13 = _mm256_load_ps((float*)&exp_c1);
    // vmovdqa ymm14,YMMWORD ptr exp_q0
    __m256 ymm14 = _mm256_load_ps((float*)&exp_q0);
    // vmovdqa ymm15,YMMWORD ptr am_1
    __m256 ymm15 = _mm256_load_ps((float*)&am_1);

    // eloop4
    while (rcx != 0) {
        // vmovaps ymm0,YMMWORD ptr [rax]
        __m256 ymm0 = _mm256_load_ps((float*)rax);
        // vminps ymm0,ymm0,ymm7
        ymm0 = _mm256_min_ps(ymm0, ymm7);
        // vmaxps ymm0,ymm0,ymm8
        ymm0 = _mm256_max_ps(ymm0, ymm8);
        // vxorps ymm2,ymm2,ymm2
        __m256 ymm2 = _mm256_setzero_ps();
        // FMA化: vfmadd213ps ymm1,ymm9,ymm10
        __m256 ymm1 = _mm256_fmadd_ps(ymm0, ymm9, ymm10);
        // vcmpnltps ymm2,ymm2,ymm1
        ymm2 = _mm256_cmp_ps(ymm2, ymm1, _CMP_NLT_US);
        // vpand ymm2,ymm2,ymm11
        ymm2 = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(ymm2), ymm11));
        // vcvttps2dq ymm1,ymm1
        __m256i ymm1_i = _mm256_cvttps_epi32(ymm1);
        // vpsubd ymm1,ymm1,ymm2
        ymm1_i = _mm256_sub_epi32(ymm1_i, _mm256_castps_si256(ymm2));
        // vcvtdq2ps ymm3,ymm1
        __m256 ymm3 = _mm256_cvtepi32_ps(ymm1_i);
        // FMA化: vfnmadd231ps ymm0,ymm3,ymm12
        ymm0 = _mm256_fnmadd_ps(ymm3, ymm12, ymm0);
        // FMA化: vfnmadd231ps ymm0,ymm3,ymm13
        ymm0 = _mm256_fnmadd_ps(ymm3, ymm13, ymm0);
        // vpaddd ymm1,ymm1,YMMWORD ptr epi32_0x7f
        ymm1_i = _mm256_add_epi32(ymm1_i, _mm256_load_si256((__m256i*)&epi32_0x7f));
        // vmovaps ymm2,ymm0
        __m256 ymm2_ps = ymm0;
        // vmulps ymm0,ymm0,ymm0
        ymm0 = _mm256_mul_ps(ymm0, ymm0);
        // FMA化: vfmadd213ps ymm6,ymm0,YMMWORD ptr exp_q1
        __m256 ymm6 = _mm256_fmadd_ps(ymm14, ymm0, _mm256_load_ps((float*)&exp_q1));
        // FMA化: vfmadd213ps ymm4,ymm0,YMMWORD ptr exp_p1
        __m256 ymm4 = _mm256_fmadd_ps(_mm256_load_ps((float*)&exp_p0), ymm0,
            _mm256_load_ps((float*)&exp_p1));
        // FMA化: vfmadd213ps ymm6,ymm0,YMMWORD ptr exp_q2
        ymm6 = _mm256_fmadd_ps(ymm6, ymm0, _mm256_load_ps((float*)&exp_q2));
        // vmulps ymm4,ymm4,ymm0
        ymm4 = _mm256_mul_ps(ymm4, ymm0);
        // FMA化: vfmadd213ps ymm6,ymm0,YMMWORD ptr exp_q3
        ymm6 = _mm256_fmadd_ps(ymm6, ymm0, _mm256_load_ps((float*)&exp_q3));
        // FMA化: vfmadd231ps ymm2,ymm4,ymm2
        ymm2_ps = _mm256_fmadd_ps(ymm4, ymm2_ps, ymm2_ps);
        // vpslld ymm1,ymm1,23
        ymm1_i = _mm256_slli_epi32(ymm1_i, 23);
        // vsubps ymm6,ymm6,ymm2
        ymm6 = _mm256_sub_ps(ymm6, ymm2_ps);
        // vdivps ymm2,ymm2,ymm6
        ymm2_ps = _mm256_div_ps(ymm2_ps, ymm6);
        // FMA化: vfmadd213ps ymm0,YMMWORD ptr exp_q3,ymm15
        ymm0 = _mm256_fmadd_ps(ymm2_ps, _mm256_load_ps((float*)&exp_q3), ymm15);
        // vmulps ymm0,ymm0,ymm1
        ymm0 = _mm256_mul_ps(ymm0, _mm256_castsi256_ps(ymm1_i));
        // vmovaps YMMWORD ptr [rax],ymm0
        _mm256_store_ps((float*)rax, ymm0);

        rax += r8;
        rcx -= rdx;
    }

    _mm256_zeroupper();
}

// 元のアセンブラ関数の引数:
// processLine0_AVX2_ASM proc tempu:dword,width_:dword,dstp:dword,src3p:dword,src_pitch:dword,val_min_max:dword
// tempu = rcx
// width_ = edx
// dstp = r8
// src3p = r9
// src_pitch = [rbp+48]
// val_min_max = [rbp+56]

extern "C" int processLine0_AVX2_ASM(
    const uint8_t* tempu, // rcx
    int width_,         // edx
    uint8_t* dstp,      // r8
    const uint8_t* src3p, // r9
    int src_pitch,      // [rbp+48]
    const uint16_t* val_min_max // [rbp+56]
) {
    // レジスタの初期化
    const char* rax = reinterpret_cast<const char*>(tempu);
    const char* rbx = reinterpret_cast<const char*>(src3p);
    int rcx = width_;
    int rdx = src_pitch;
    char* rsi = reinterpret_cast<char*>(dstp);
    const int r8 = 32;
    const char* r10 = reinterpret_cast<const char*>(val_min_max);

    // ポインタの計算
    const char* rdi = rbx + rdx * 4;

    // 定数のロード
    // vmovdqa ymm8,YMMWORD ptr w_19
    __m256i ymm8 = _mm256_load_si256((__m256i*)&w_19);
    // vmovdqa ymm9,YMMWORD ptr w_3
    __m256i ymm9 = _mm256_load_si256((__m256i*)&w_3);
    // vmovdqa ymm10,YMMWORD ptr ub_1
    __m256i ymm10 = _mm256_load_si256((__m256i*)&ub_1);
    // vmovdqa ymm11,YMMWORD ptr uw_16
    __m256i ymm11 = _mm256_load_si256((__m256i*)&uw_16);
    // vmovdqa ymm12,YMMWORD ptr[r10]
    __m256i ymm12 = _mm256_load_si256((__m256i*)r10);
    // vmovdqa ymm13,YMMWORD ptr[r10+64]
    __m256i ymm13 = _mm256_load_si256((__m256i*)(r10 + 64));
    // vpxor ymm6,ymm6,ymm6
    __m256i ymm6 = _mm256_setzero_si256();
    // vpxor ymm7,ymm7,ymm7
    __m256i ymm7 = _mm256_setzero_si256();

    // xloop
    while (rcx != 0) {
        // vmovdqa ymm0,YMMWORD PTR [rbx+rdx*2]
        __m256i ymm0 = _mm256_load_si256((__m256i*)(rbx + rdx * 2));
        // vmovdqa ymm1,YMMWORD PTR [rdi]
        __m256i ymm1 = _mm256_load_si256((__m256i*)rdi);
        // vpunpckhbw ymm2,ymm0,ymm7
        __m256i ymm2 = _mm256_unpackhi_epi8(ymm0, ymm7);
        // vpunpckhbw ymm3,ymm1,ymm7
        __m256i ymm3 = _mm256_unpackhi_epi8(ymm1, ymm7);
        // vpunpcklbw ymm0,ymm0,ymm7
        ymm0 = _mm256_unpacklo_epi8(ymm0, ymm7);
        // vpunpcklbw ymm1,ymm1,ymm7
        ymm1 = _mm256_unpacklo_epi8(ymm1, ymm7);
        // vpaddw ymm0,ymm0,ymm1
        ymm0 = _mm256_add_epi16(ymm0, ymm1);
        // vpaddw ymm2,ymm2,ymm3
        ymm2 = _mm256_add_epi16(ymm2, ymm3);
        // vpmullw ymm0,ymm0,ymm8
        ymm0 = _mm256_mullo_epi16(ymm0, ymm8);
        // vpmullw ymm2,ymm2,ymm8
        ymm2 = _mm256_mullo_epi16(ymm2, ymm8);
        // vmovdqa ymm1,YMMWORD PTR [rbx]
        ymm1 = _mm256_load_si256((__m256i*)rbx);
        // vmovdqa ymm3,YMMWORD PTR [rdi+rdx*2]
        ymm3 = _mm256_load_si256((__m256i*)(rdi + rdx * 2));
        // vpunpckhbw ymm4,ymm1,ymm7
        __m256i ymm4 = _mm256_unpackhi_epi8(ymm1, ymm7);
        // vpunpckhbw ymm5,ymm3,ymm7
        __m256i ymm5 = _mm256_unpackhi_epi8(ymm3, ymm7);
        // vpunpcklbw ymm1,ymm1,ymm7
        ymm1 = _mm256_unpacklo_epi8(ymm1, ymm7);
        // vpunpcklbw ymm3,ymm3,ymm7
        ymm3 = _mm256_unpacklo_epi8(ymm3, ymm7);
        // vpaddw ymm1,ymm1,ymm3
        ymm1 = _mm256_add_epi16(ymm1, ymm3);
        // vpaddw ymm4,ymm4,ymm5
        ymm4 = _mm256_add_epi16(ymm4, ymm5);
        // vpmullw ymm1,ymm1,ymm9
        ymm1 = _mm256_mullo_epi16(ymm1, ymm9);
        // vpmullw ymm4,ymm4,ymm9
        ymm4 = _mm256_mullo_epi16(ymm4, ymm9);
        // vmovdqa ymm5,YMMWORD PTR [rax]
        ymm5 = _mm256_load_si256((__m256i*)rax);
        // vpsubusw ymm0,ymm0,ymm1
        ymm0 = _mm256_subs_epu16(ymm0, ymm1);
        // vpsubusw ymm2,ymm2,ymm4
        ymm2 = _mm256_subs_epu16(ymm2, ymm4);
        // vpxor ymm5,ymm5,ymm10
        ymm5 = _mm256_xor_si256(ymm5, ymm10);
        // vpaddusw ymm0,ymm0,ymm11
        ymm0 = _mm256_adds_epu16(ymm0, ymm11);
        // vpaddusw ymm2,ymm2,ymm11
        ymm2 = _mm256_adds_epu16(ymm2, ymm11);
        // vpsadbw ymm5,ymm5,ymm7
        ymm5 = _mm256_sad_epu8(ymm5, ymm7);
        // vpsraw ymm0,ymm0,5
        ymm0 = _mm256_srai_epi16(ymm0, 5);
        // vpsraw ymm2,ymm2,5
        ymm2 = _mm256_srai_epi16(ymm2, 5);
        // vmovdqa ymm3,ymm5
        ymm3 = ymm5;
        // vpminsw ymm0,ymm0,ymm13
        ymm0 = _mm256_min_epi16(ymm0, ymm13);
        // vpsrldq ymm5,ymm5,8
        ymm5 = _mm256_srli_si256(ymm5, 8);
        // vpminsw ymm2,ymm2,ymm13
        ymm2 = _mm256_min_epi16(ymm2, ymm13);
        // vpaddusw ymm5,ymm5,ymm3
        ymm5 = _mm256_adds_epu16(ymm5, ymm3);
        // vpmaxsw ymm0,ymm0,ymm12
        ymm0 = _mm256_max_epi16(ymm0, ymm12);
        // vpmaxsw ymm2,ymm2,ymm12
        ymm2 = _mm256_max_epi16(ymm2, ymm12);
        // vextracti128 xmm3,ymm5,1
        __m128i xmm3 = _mm256_extracti128_si256(ymm5, 1);
        // vpackuswb ymm0,ymm0,ymm2
        ymm0 = _mm256_packus_epi16(ymm0, ymm2);
        // vpaddusw xmm5,xmm5,xmm3
        __m128i xmm5 = _mm_adds_epu16(_mm256_castsi256_si128(ymm5), xmm3);
        // vmovdqa YMMWORD PTR [rsi],ymm0
        _mm256_store_si256((__m256i*)rsi, ymm0);
        // vpaddusw xmm6,xmm6,xmm5
        ymm6 = _mm256_inserti128_si256(ymm6, _mm_adds_epu16(_mm256_castsi256_si128(ymm6), xmm5), 0);

        rbx += r8;
        rdi += r8;
        rax += r8;
        rsi += r8;
        rcx -= r8;
    }

    // xor rax,rax
    int ret = 0;
    // vmovd eax,xmm6
    ret = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm6));

    _mm256_zeroupper();
    return ret;
}

// 元のアセンブラ関数の引数:
// processLine0_AVX2_ASM_16 proc tempu:dword,width_:dword,dstp:dword,src3p:dword,src_pitch:dword,val_min_max:dword
// tempu = rcx
// width_ = edx
// dstp = r8
// src3p = r9
// src_pitch = [rbp+48]
// val_min_max = [rbp+56]

extern "C" int processLine0_AVX2_ASM_16(
    const uint8_t* tempu, // rcx
    int width_,         // edx
    uint8_t* dstp,      // r8
    const uint8_t* src3p, // r9
    int src_pitch,      // [rbp+48]
    const uint16_t* val_min_max // [rbp+56]
) {
    // レジスタの初期化
    const char* rax = reinterpret_cast<const char*>(tempu);
    const char* rbx = reinterpret_cast<const char*>(src3p);
    int rcx = width_;
    int rdx = src_pitch;
    char* rsi = reinterpret_cast<char*>(dstp);
    const int r8 = 32;
    const int r9 = 16;
    const char* r10 = reinterpret_cast<const char*>(val_min_max);

    // ポインタの計算
    const char* rdi = rbx + rdx * 4;

    // 定数のロード
    // vmovdqa ymm8,YMMWORD ptr d_19
    __m256i ymm8 = _mm256_load_si256((__m256i*)&d_19);
    // vmovdqa ymm9,YMMWORD ptr d_3
    __m256i ymm9 = _mm256_load_si256((__m256i*)&d_3);
    // vmovdqa xmm10,XMMWORD ptr ub_1
    __m128i xmm10 = _mm_load_si128((__m128i*)&ub_1);
    // vmovdqa ymm11,YMMWORD ptr ud_16
    __m256i ymm11 = _mm256_load_si256((__m256i*)&ud_16);
    // vmovdqa ymm12,YMMWORD ptr[r10]
    __m256i ymm12 = _mm256_load_si256((__m256i*)r10);
    // vmovdqa ymm13,YMMWORD ptr[r10+64]
    __m256i ymm13 = _mm256_load_si256((__m256i*)(r10 + 64));
    // vpxor ymm6,ymm6,ymm6
    __m256i ymm6 = _mm256_setzero_si256();
    // vpxor ymm7,ymm7,ymm7
    __m256i ymm7 = _mm256_setzero_si256();

    // xloop_16
    while (rcx != 0) {
        // vmovdqa ymm0,YMMWORD ptr[rbx+rdx*2]
        __m256i ymm0 = _mm256_load_si256((__m256i*)(rbx + rdx * 2));
        // vmovdqa ymm1,YMMWORD ptr[rdi]
        __m256i ymm1 = _mm256_load_si256((__m256i*)rdi);
        // vpunpckhwd ymm2,ymm0,ymm7
        __m256i ymm2 = _mm256_unpackhi_epi16(ymm0, ymm7);
        // vpunpckhwd ymm3,ymm1,ymm7
        __m256i ymm3 = _mm256_unpackhi_epi16(ymm1, ymm7);
        // vpunpcklwd ymm0,ymm0,ymm7
        ymm0 = _mm256_unpacklo_epi16(ymm0, ymm7);
        // vpunpcklwd ymm1,ymm1,ymm7
        ymm1 = _mm256_unpacklo_epi16(ymm1, ymm7);
        // vpaddd ymm0,ymm0,ymm1
        ymm0 = _mm256_add_epi32(ymm0, ymm1);
        // vpaddd ymm2,ymm2,ymm3
        ymm2 = _mm256_add_epi32(ymm2, ymm3);
        // vpmulld ymm0,ymm0,ymm8
        ymm0 = _mm256_mullo_epi32(ymm0, ymm8);
        // vpmulld ymm2,ymm2,ymm8
        ymm2 = _mm256_mullo_epi32(ymm2, ymm8);
        // vmovdqa ymm1,YMMWORD ptr[rbx]
        ymm1 = _mm256_load_si256((__m256i*)rbx);
        // vmovdqa ymm3,YMMWORD ptr[rdi+rdx*2]
        ymm3 = _mm256_load_si256((__m256i*)(rdi + rdx * 2));
        // vpunpckhwd ymm4,ymm1,ymm7
        __m256i ymm4 = _mm256_unpackhi_epi16(ymm1, ymm7);
        // vpunpckhwd ymm5,ymm3,ymm7
        __m256i ymm5 = _mm256_unpackhi_epi16(ymm3, ymm7);
        // vpunpcklwd ymm1,ymm1,ymm7
        ymm1 = _mm256_unpacklo_epi16(ymm1, ymm7);
        // vpunpcklwd ymm3,ymm3,ymm7
        ymm3 = _mm256_unpacklo_epi16(ymm3, ymm7);
        // vpaddd ymm1,ymm1,ymm3
        ymm1 = _mm256_add_epi32(ymm1, ymm3);
        // vpaddd ymm4,ymm4,ymm5
        ymm4 = _mm256_add_epi32(ymm4, ymm5);
        // vpmulld ymm1,ymm1,ymm9
        ymm1 = _mm256_mullo_epi32(ymm1, ymm9);
        // vpmulld ymm4,ymm4,ymm9
        ymm4 = _mm256_mullo_epi32(ymm4, ymm9);
        // vpsubd ymm0,ymm0,ymm1
        ymm0 = _mm256_sub_epi32(ymm0, ymm1);
        // vpsubd ymm2,ymm2,ymm4
        ymm2 = _mm256_sub_epi32(ymm2, ymm4);
        // vmovdqa xmm5,XMMWORD ptr [rax]
        __m128i xmm5 = _mm_load_si128((__m128i*)rax);
        // vpaddd ymm0,ymm0,ymm11
        ymm0 = _mm256_add_epi32(ymm0, ymm11);
        // vpaddd ymm2,ymm2,ymm11
        ymm2 = _mm256_add_epi32(ymm2, ymm11);
        // vpxor xmm5,xmm5,xmm10
        xmm5 = _mm_xor_si128(xmm5, xmm10);
        // vpsrad ymm0,ymm0,5
        ymm0 = _mm256_srai_epi32(ymm0, 5);
        // vpsrad ymm2,ymm2,5
        ymm2 = _mm256_srai_epi32(ymm2, 5);
        // vpsadbw xmm5,xmm5,xmm7
        xmm5 = _mm_sad_epu8(xmm5, _mm256_castsi256_si128(ymm7));
        // vpackusdw ymm0,ymm0,ymm2
        ymm0 = _mm256_packus_epi32(ymm0, ymm2);
        // vmovdqa xmm3,xmm5
        __m128i xmm3 = xmm5;
        // vpminuw ymm0,ymm0,ymm13
        ymm0 = _mm256_min_epu16(ymm0, ymm13);
        // vpsrldq xmm5,xmm5,8
        xmm5 = _mm_srli_si128(xmm5, 8);
        // vpmaxuw ymm0,ymm0,ymm12
        ymm0 = _mm256_max_epu16(ymm0, ymm12);
        // vpaddusw xmm5,xmm5,xmm3
        xmm5 = _mm_adds_epu16(xmm5, xmm3);
        // vmovdqa YMMWORD ptr [rsi],ymm0
        _mm256_store_si256((__m256i*)rsi, ymm0);
        // vpaddusw xmm6,xmm6,xmm5
        ymm6 = _mm256_inserti128_si256(ymm6, _mm_adds_epu16(_mm256_castsi256_si128(ymm6), xmm5), 0);

        rbx += r8;
        rdi += r8;
        rax += r9;
        rsi += r8;
        rcx -= r9;
    }

    // xor rax,rax
    int ret = 0;
    // vmovd eax,xmm6
    ret = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm6));

    _mm256_zeroupper();
    return ret;
}

// 元のアセンブラ関数の引数:
// processLine0_AVX2_ASM_32 proc tempu:dword,width_:dword,dstp:dword,src3p:dword,src_pitch:dword
// tempu = rcx
// width_ = edx
// dstp = r8
// src3p = r9
// src_pitch = [rbp+48]

extern "C" int processLine0_AVX2_ASM_32(
    const uint8_t* tempu, // rcx
    int width_,         // edx
    uint8_t* dstp,      // r8
    const uint8_t* src3p, // r9
    int src_pitch       // [rbp+48]
) {
    // レジスタの初期化
    const char* rax = reinterpret_cast<const char*>(tempu);
    const char* rbx = reinterpret_cast<const char*>(src3p);
    int rcx = width_;
    int rdx = src_pitch;
    char* rsi = reinterpret_cast<char*>(dstp);
    const int r8 = 32;
    const int r9 = 8;

    // ポインタの計算
    const char* rdi = rbx + rdx * 4;

    // レジスタの初期化
    // vpxor ymm5,ymm5,ymm5
    __m256i ymm5 = _mm256_setzero_si256();
    // vpxor ymm6,ymm6,ymm6
    __m256i ymm6 = _mm256_setzero_si256();
    // vmovaps ymm7,YMMWORD ptr f_19
    __m256 ymm7 = _mm256_load_ps((float*)&f_19);
    // vmovaps ymm8,YMMWORD ptr f_3
    __m256 ymm8 = _mm256_load_ps((float*)&f_3);
    // vmovdqa xmm9,XMMWORD ptr uw_1
    __m128i xmm9 = _mm_load_si128((__m128i*)&uw_1);

    // xloop_32
    while (rcx != 0) {
        // vmovq xmm4,qword ptr [rax]
        __m128i xmm4 = _mm_loadl_epi64((__m128i*)rax);
        // vmovaps ymm2,YMMWORD ptr[rbx]
        __m256 ymm2 = _mm256_load_ps((float*)rbx);
        // vmovaps ymm0,YMMWORD ptr[rbx+rdx*2]
        __m256 ymm0 = _mm256_load_ps((float*)(rbx + rdx * 2));
        // vpunpcklbw xmm4,xmm4,xmm6
        xmm4 = _mm_unpacklo_epi8(xmm4, _mm256_castsi256_si128(ymm6));
        // vmovaps ymm1,YMMWORD ptr[rdi]
        __m256 ymm1 = _mm256_load_ps((float*)rdi);
        // vmovaps ymm3,YMMWORD ptr[rdi+rdx*2]
        __m256 ymm3 = _mm256_load_ps((float*)(rdi + rdx * 2));
        // vaddps ymm0,ymm0,ymm1
        ymm0 = _mm256_add_ps(ymm0, ymm1);
        // vpxor xmm4,xmm4,xmm9
        xmm4 = _mm_xor_si128(xmm4, xmm9);
        // vaddps ymm2,ymm2,ymm3
        ymm2 = _mm256_add_ps(ymm2, ymm3);
        // vpsadbw xmm4,xmm4,xmm6
        xmm4 = _mm_sad_epu8(xmm4, _mm256_castsi256_si128(ymm6));
        // vmulps ymm0,ymm0,ymm7
        ymm0 = _mm256_mul_ps(ymm0, ymm7);
        // vmovdqa xmm3,xmm4
        __m128i xmm3 = xmm4;
        // vmulps ymm2,ymm2,ymm8
        ymm2 = _mm256_mul_ps(ymm2, ymm8);
        // vpsrldq xmm4,xmm4,8
        xmm4 = _mm_srli_si128(xmm4, 8);
        // vsubps ymm0,ymm0,ymm2
        ymm0 = _mm256_sub_ps(ymm0, ymm2);
        // vpaddusw xmm4,xmm4,xmm3
        xmm4 = _mm_adds_epu16(xmm4, xmm3);
        // vmovaps YMMWORD ptr[rsi],ymm0
        _mm256_store_ps((float*)rsi, ymm0);
        // vpaddusw xmm5,xmm5,xmm4
        ymm5 = _mm256_inserti128_si256(ymm5, _mm_adds_epu16(_mm256_castsi256_si128(ymm5), xmm4), 0);

        rbx += r8;
        rdi += r8;
        rax += r9;
        rsi += r8;
        rcx -= r9;
    }

    // xor rax,rax
    int ret = 0;
    // vmovd eax,xmm5
    ret = _mm_cvtsi128_si32(_mm256_castsi256_si128(ymm5));

    _mm256_zeroupper();
    return ret;
}

// 元のアセンブラ関数の引数:
// weightedAvgElliottMul5_m16_FMA3 proc ptr_w:dword,n:dword,mstd:dword
// ptr_w = rcx
// n = edx
// mstd = r8

extern "C" void weightedAvgElliottMul5_m16_FMA3(
    const float* ptr_w, // rcx
    int n,          // edx
    float* mstd     // r8
) {
    // レジスタの初期化
    const char* rax = reinterpret_cast<const char*>(ptr_w);
    int rcx = n;
    const int r9 = 16;

    // 定数のロード
    // vmovdqa ymm6,YMMWORD ptr sign_bits_f_32
    __m256 ymm6 = _mm256_load_ps((float*)&sign_bits_f_32);
    // vmovdqa ymm7,YMMWORD ptr ones_f_32
    __m256 ymm7 = _mm256_load_ps((float*)&ones_f_32);

    // ポインタの計算
    const char* rdx = rax + rcx * 4;
    int rdi = 0;

    // レジスタの初期化
    // AVX-512版と同じ16 laneの加算木に揃えるため、low/highを別々に積算する。
    __m256 weight_low = _mm256_setzero_ps();
    __m256 weight_high = _mm256_setzero_ps();
    __m256 value_low = _mm256_setzero_ps();
    __m256 value_high = _mm256_setzero_ps();

    // nloop_52
    while (rcx != 0) {
        const __m256 low_weight = _mm256_load_ps((float*)(rax + rdi * 4));
        const __m256 low_output = _mm256_load_ps((float*)(rdx + rdi * 4));
        const __m256 low_denominator = _mm256_add_ps(_mm256_and_ps(low_output, ymm6), ymm7);
        const __m256 low_elliott = _mm256_div_ps(low_output, low_denominator);
        weight_low = _mm256_add_ps(weight_low, low_weight);
        value_low = _mm256_fmadd_ps(low_weight, low_elliott, value_low);

        const __m256 high_weight = _mm256_load_ps((float*)(rax + rdi * 4 + 32));
        const __m256 high_output = _mm256_load_ps((float*)(rdx + rdi * 4 + 32));
        const __m256 high_denominator = _mm256_add_ps(_mm256_and_ps(high_output, ymm6), ymm7);
        const __m256 high_elliott = _mm256_div_ps(high_output, high_denominator);
        weight_high = _mm256_add_ps(weight_high, high_weight);
        value_high = _mm256_fmadd_ps(high_weight, high_elliott, value_high);

        rdi += r9;
        rcx -= r9;
    }

    const __m256 weight_sum = _mm256_add_ps(weight_low, weight_high);
    const __m256 value_sum = _mm256_add_ps(value_low, value_high);
    __m128 xmm0 = _mm_add_ps(
        _mm256_castps256_ps128(weight_sum), _mm256_extractf128_ps(weight_sum, 1));
    __m128 xmm1 = _mm_add_ps(
        _mm256_castps256_ps128(value_sum), _mm256_extractf128_ps(value_sum, 1));
    xmm0 = _mm_hadd_ps(xmm0, xmm0);
    xmm0 = _mm_hadd_ps(xmm0, xmm0);
    xmm1 = _mm_hadd_ps(xmm1, xmm1);
    xmm1 = _mm_hadd_ps(xmm1, xmm1);

    // vcomiss xmm0,dword ptr min_weight_sum
    if (!(_mm_cvtss_f32(xmm0) > _mm_cvtss_f32(_mm_load_ss((float*)&min_weight_sum)))) {
        // nodiv2:
        // vxorps xmm1,xmm1,xmm1
        xmm1 = _mm_setzero_ps();
    } else {
        // vmulss xmm1,xmm1,dword ptr five_f
        xmm1 = _mm_mul_ss(xmm1, _mm_load_ss((float*)&five_f));
        // vdivss xmm1,xmm1,xmm0
        xmm1 = _mm_div_ss(xmm1, xmm0);
    }

    // finish_52:
    // FMA化: vfmadd213ss xmm1,dword ptr[r8+4],dword ptr[r8]
    xmm1 = _mm_fmadd_ss(xmm1, _mm_load_ss(mstd + 1), _mm_load_ss(mstd));
    // vaddss xmm1,xmm1,dword ptr[r8+12]
    xmm1 = _mm_add_ss(xmm1, _mm_load_ss(mstd + 3));
    // vmovss dword ptr[r8+12],xmm1
    _mm_store_ss(mstd + 3, xmm1);
}

// 元のアセンブラ関数の引数:
// extract_m8_FMA3 proc srcp:dword,stride:dword,xdia:dword,ydia:dword,mstd:dword,input:dword
// srcp = rcx
// stride = edx
// xdia = r8d
// ydia = r9d
// mstd = [rbp+48]
// input = [rbp+56]

extern "C" void extract_m8_FMA3(
    const uint8_t* srcp, // rcx
    int stride,     // edx
    int xdia,       // r8d
    int ydia,       // r9d
    float* mstd,    // [rbp+48]
    float* input    // [rbp+56]
) {
    // レジスタの初期化
    const char* rax = reinterpret_cast<const char*>(srcp);
    int rbx = stride;
    int rdi = xdia;
    char* rsi = reinterpret_cast<char*>(input);
    int r8 = ydia;
    const int r10 = 2;
    const int r11 = 8;
    const int r12 = 32;

    // ポインタの計算
    const char* rdx = rax + rbx * 2;

    // レジスタの初期化
    // vpxor ymm5,ymm5,ymm5
    __m256 ymm5 = _mm256_setzero_ps();
    // vpxor ymm6,ymm6,ymm6
    __m256 ymm6 = _mm256_setzero_ps();
    // vpxor ymm4,ymm4,ymm4
    __m256 ymm4 = _mm256_setzero_ps();

    // yloop2a
    while (r8 != 0) {
        int rcx = 0;
        // xloop2a
        while (rcx < rdi) {
            // vmovq xmm0,QWORD PTR[rax+rcx]
            __m128i xmm0 = _mm_loadl_epi64((__m128i*)(rax + rcx));
            // vmovq xmm2,QWORD PTR[rdx+rcx]
            __m128i xmm2 = _mm_loadl_epi64((__m128i*)(rdx + rcx));
            // vpunpcklbw xmm0,xmm0,xmm4
            xmm0 = _mm_unpacklo_epi8(xmm0, _mm256_castsi256_si128(_mm256_castps_si256(ymm4)));
            // vpunpcklbw xmm2,xmm2,xmm4
            xmm2 = _mm_unpacklo_epi8(xmm2, _mm256_castsi256_si128(_mm256_castps_si256(ymm4)));
            // vmovhlps xmm1,xmm4,xmm0
            __m128 xmm1 = _mm_movehl_ps(_mm256_castps256_ps128(ymm4), _mm_castsi128_ps(xmm0));
            // vmovhlps xmm3,xmm4,xmm2
            __m128 xmm3 = _mm_movehl_ps(_mm256_castps256_ps128(ymm4), _mm_castsi128_ps(xmm2));
            // vinserti128 ymm0,ymm0,xmm1,1
            __m256i ymm0_i = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm0), _mm_castps_si128(xmm1), 1);
            // vinserti128 ymm2,ymm2,xmm3,1
            __m256i ymm2_i = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm2), _mm_castps_si128(xmm3), 1);
            // vpunpcklwd ymm0,ymm0,ymm4
            ymm0_i = _mm256_unpacklo_epi16(ymm0_i, _mm256_castps_si256(ymm4));
            // vpunpcklwd ymm2,ymm2,ymm4
            ymm2_i = _mm256_unpacklo_epi16(ymm2_i, _mm256_castps_si256(ymm4));
            // vcvtdq2ps ymm0,ymm0
            __m256 ymm0 = _mm256_cvtepi32_ps(ymm0_i);
            // vcvtdq2ps ymm2,ymm2
            __m256 ymm2 = _mm256_cvtepi32_ps(ymm2_i);
            // vmovaps YMMWORD PTR[rsi],ymm0
            _mm256_store_ps((float*)rsi, ymm0);
            // vmovaps YMMWORD PTR[rsi+rdi*4],ymm2
            _mm256_store_ps((float*)(rsi + rdi * 4), ymm2);
            // vaddps ymm5,ymm5,ymm0
            ymm5 = _mm256_add_ps(ymm5, ymm0);
            // vaddps ymm5,ymm5,ymm2
            ymm5 = _mm256_add_ps(ymm5, ymm2);
            // vfmadd231ps ymm6,ymm0,ymm0
            ymm6 = _mm256_fmadd_ps(ymm0, ymm0, ymm6);
            // vfmadd231ps ymm6,ymm2,ymm2
            ymm6 = _mm256_fmadd_ps(ymm2, ymm2, ymm6);

            rcx += r11;
            rsi += r12;
        }

        rax += rbx * 4;
        rdx += rbx * 4;
        rsi += rdi * 4;
        r8 -= r10;
    }

    // vextractf128 xmm0,ymm5,1
    __m128 xmm0 = _mm256_extractf128_ps(ymm5, 1);
    // vextractf128 xmm2,ymm6,1
    __m128 xmm2 = _mm256_extractf128_ps(ymm6, 1);
    // vaddps xmm5,xmm5,xmm0
    __m128 xmm5 = _mm_add_ps(_mm256_castps256_ps128(ymm5), xmm0);
    // vaddps xmm6,xmm6,xmm2
    __m128 xmm6 = _mm_add_ps(_mm256_castps256_ps128(ymm6), xmm2);

    // mov eax,r9d
    int eax = ydia;
    // vmovhlps xmm0,xmm0,xmm5
    xmm0 = _mm_movehl_ps(xmm0, xmm5);
    // vmovhlps xmm1,xmm1,xmm6
    __m128 xmm1 = _mm_movehl_ps(xmm6, xmm6);
    // mul edi
    eax *= rdi;
    // vaddps xmm5,xmm5,xmm0
    xmm5 = _mm_add_ps(xmm5, xmm0);
    // vaddps xmm6,xmm6,xmm1
    xmm6 = _mm_add_ps(xmm6, xmm1);
    // vcvtsi2ss xmm7,xmm7,eax
    __m128 xmm7 = _mm_cvtsi32_ss(_mm_setzero_ps(), eax);
    // vpshuflw xmm0,xmm5,14
    xmm0 = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(xmm5), 14));
    // vpshuflw xmm1,xmm6,14
    xmm1 = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(xmm6), 14));
    // vrcpss xmm7,xmm7,xmm7
    xmm7 = _mm_rcp_ss(xmm7);
    // vaddss xmm5,xmm5,xmm0
    xmm5 = _mm_add_ss(xmm5, xmm0);
    // vaddss xmm6,xmm6,xmm1
    xmm6 = _mm_add_ss(xmm6, xmm1);
    // mov rax,mstd
    char* rax_ptr = reinterpret_cast<char*>(mstd);
    // vmulss xmm5,xmm5,xmm7
    xmm5 = _mm_mul_ss(xmm5, xmm7);
    // vmulss xmm6,xmm6,xmm7
    xmm6 = _mm_mul_ss(xmm6, xmm7);
    // vmovss dword ptr[rax],xmm5
    _mm_store_ss((float*)rax_ptr, xmm5);
    // vmulss xmm5,xmm5,xmm5
    xmm5 = _mm_mul_ss(xmm5, xmm5);
    // vsubss xmm6,xmm6,xmm5
    xmm6 = _mm_sub_ss(xmm6, xmm5);
    // vcomiss xmm6,dword ptr flt_epsilon_sse
    if (_mm_comile_ss(xmm6, _mm_load_ss((float*)&flt_epsilon_sse))) {
        // novarjmpa:
        // vmovss dword ptr[rax+4],xmm4
        _mm_store_ss((float*)(rax_ptr + 4), _mm256_castps256_ps128(ymm4));
        // vmovss dword ptr[rax+8],xmm4
        _mm_store_ss((float*)(rax_ptr + 8), _mm256_castps256_ps128(ymm4));
    } else {
        // vrsqrtss xmm6,xmm6,xmm6
        xmm6 = _mm_rsqrt_ss(xmm6);
        // vrcpss xmm5,xmm5,xmm6
        xmm5 = _mm_rcp_ss(xmm6);
        // vmovss dword ptr[rax+4],xmm5
        _mm_store_ss((float*)(rax_ptr + 4), xmm5);
        // vmovss dword ptr[rax+8],xmm6
        _mm_store_ss((float*)(rax_ptr + 8), xmm6);
    }

    // finish_3a:
    // vmovss dword ptr[rax+12],xmm4
    _mm_store_ss((float*)(rax_ptr + 12), _mm256_castps256_ps128(ymm4));

    _mm256_zeroupper();
}


// 元のアセンブラ関数の引数:
// extract_m8_i16_AVX2 proc srcp:dword,stride:dword,xdia:dword,ydia:dword,mstd:dword,inputf:dword
// srcp  = rcx
// stride = edx
// xdia  = r8d
// ydia  = r9d
// mstd  = [rbp+48]
// inputf= [rbp+56]
extern "C" void extract_m8_i16_AVX2(
    const uint8_t* srcp, // rcx
    int stride,     // edx
    int xdia,       // r8d
    int ydia,       // r9d
    float* mstd,    // [rbp+48]
    float* inputf   // [rbp+56]
) {
    // レジスタ変数の定義
    const char* rax_ptr = reinterpret_cast<const char*>(srcp);
    int rbx = stride;
    int rdi = xdia;
    char* rdx_ptr = reinterpret_cast<char*>(inputf);
    int r8 = ydia;
    int r10 = 2;
    int r11 = 16;
    int r12 = 32;

    // スタックフレームの設定
    __m256i ymm4 = _mm256_setzero_si256();
    __m256i ymm5 = _mm256_setzero_si256();
    __m256i ymm6 = _mm256_setzero_si256();

    // メインループ
    if (rdi <= 8) {
        // yloop_ ループ
        for (int i = 0; i < r8; i += r10) {
            // vmovq xmm2,QWORD PTR[rax]
            __m128i xmm2 = _mm_loadl_epi64((__m128i*)rax_ptr);
            // vmovq xmm3,QWORD PTR[rsi]
            __m128i xmm3 = _mm_loadl_epi64((__m128i*)(rax_ptr + rbx * 2));
            // vpunpcklbw xmm0,xmm2,xmm6
            __m128i xmm0 = _mm_unpacklo_epi8(xmm2, _mm256_castsi256_si128(ymm6));
            // vpunpcklbw xmm1,xmm3,xmm6
            __m128i xmm1 = _mm_unpacklo_epi8(xmm3, _mm256_castsi256_si128(ymm6));
            // vpsadbw xmm2,xmm2,xmm6
            xmm2 = _mm_sad_epu8(xmm2, _mm256_castsi256_si128(ymm6));
            // vpsadbw xmm3,xmm3,xmm6
            xmm3 = _mm_sad_epu8(xmm3, _mm256_castsi256_si128(ymm6));
            // vmovdqa XMMWORD ptr [rdx],xmm0
            _mm_store_si128((__m128i*)rdx_ptr, xmm0);
            // vmovdqa XMMWORD ptr [rdx+rdi*2],xmm1
            _mm_store_si128((__m128i*)(rdx_ptr + rdi * 2), xmm1);
            // vpmaddwd xmm0,xmm0,xmm0
            xmm0 = _mm_madd_epi16(xmm0, xmm0);
            // vpmaddwd xmm1,xmm1,xmm1
            xmm1 = _mm_madd_epi16(xmm1, xmm1);
            // vpaddd xmm4,xmm4,xmm2
            ymm4 = _mm256_add_epi32(ymm4, _mm256_castsi128_si256(xmm2));
            // vpaddd xmm5,xmm5,xmm0
            ymm5 = _mm256_add_epi32(ymm5, _mm256_castsi128_si256(xmm0));
            // vpaddd xmm4,xmm4,xmm3
            ymm4 = _mm256_add_epi32(ymm4, _mm256_castsi128_si256(xmm3));
            // vpaddd xmm5,xmm5,xmm1
            ymm5 = _mm256_add_epi32(ymm5, _mm256_castsi128_si256(xmm1));

            rdx_ptr += r11;
            rax_ptr += rbx * 4;
            rdx_ptr += rdi * 2;
        }
    } else {
        // アライメントチェック
        if ((uintptr_t)rax_ptr & 15) {
            // yloop__ ループ
            for (int i = 0; i < r8; i += r10) {
                for (int j = 0; j < rdi; j += r11) {
                    // vmovdqu xmm2,XMMWORD PTR[rax+rcx]
                    __m128i xmm2 = _mm_loadu_si128((__m128i*)(rax_ptr + j));
                    // vmovdqu xmm3,XMMWORD PTR[rsi+rcx]
                    __m128i xmm3 = _mm_loadu_si128((__m128i*)(rax_ptr + rbx * 2 + j));
                    // vmovhlps xmm0,xmm6,xmm2
                    __m128i xmm0 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm6)), _mm_castsi128_ps(xmm2)));
                    // vmovhlps xmm1,xmm6,xmm3
                    __m128i xmm1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm6)), _mm_castsi128_ps(xmm3)));
                    // vinserti128 ymm2,ymm2,xmm0,1
                    __m256i ymm2 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm2), xmm0, 1);
                    // vinserti128 ymm3,ymm3,xmm1,1
                    __m256i ymm3 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm3), xmm1, 1);
                    // vpunpcklbw ymm0,ymm2,ymm6
                    __m256i ymm0 = _mm256_unpacklo_epi8(ymm2, ymm6);
                    // vpunpcklbw ymm1,ymm3,ymm6
                    __m256i ymm1 = _mm256_unpacklo_epi8(ymm3, ymm6);
                    // vpsadbw xmm2,xmm2,xmm6
                    xmm2 = _mm_sad_epu8(_mm256_castsi256_si128(ymm2), _mm256_castsi256_si128(ymm6));
                    // vpsadbw xmm3,xmm3,xmm6
                    xmm3 = _mm_sad_epu8(_mm256_castsi256_si128(ymm3), _mm256_castsi256_si128(ymm6));
                    // vmovdqa YMMWORD PTR[rdx],ymm0
                    _mm256_store_si256((__m256i*)rdx_ptr, ymm0);
                    // vmovdqa YMMWORD PTR[rdx+rdi*2],ymm1
                    _mm256_store_si256((__m256i*)(rdx_ptr + rdi * 2), ymm1);
                    // vpmaddwd ymm0,ymm0,ymm0
                    ymm0 = _mm256_madd_epi16(ymm0, ymm0);
                    // vpmaddwd ymm1,ymm1,ymm1
                    ymm1 = _mm256_madd_epi16(ymm1, ymm1);
                    // vpaddd xmm4,xmm4,xmm2
                    ymm4 = _mm256_add_epi32(ymm4, _mm256_castsi128_si256(xmm2));
                    // vpaddd ymm5,ymm5,ymm0
                    ymm5 = _mm256_add_epi32(ymm5, ymm0);
                    // vpaddd xmm4,xmm4,xmm3
                    ymm4 = _mm256_add_epi32(ymm4, _mm256_castsi128_si256(xmm3));
                    // vpaddd ymm5,ymm5,ymm1
                    ymm5 = _mm256_add_epi32(ymm5, ymm1);

                    rdx_ptr += r12;
                }
                rax_ptr += rbx * 4;
                rdx_ptr += rdi * 2;
            }
        } else {
            // yloop ループ
            for (int i = 0; i < r8; i += r10) {
                for (int j = 0; j < rdi; j += r11) {
                    // vmovdqa xmm2,XMMWORD PTR[rax+rcx]
                    __m128i xmm2 = _mm_load_si128((__m128i*)(rax_ptr + j));
                    // vmovdqa xmm3,XMMWORD PTR[rsi+rcx]
                    __m128i xmm3 = _mm_load_si128((__m128i*)(rax_ptr + rbx * 2 + j));
                    // vmovhlps xmm0,xmm6,xmm2
                    __m128i xmm0 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm6)), _mm_castsi128_ps(xmm2)));
                    // vmovhlps xmm1,xmm6,xmm3
                    __m128i xmm1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm6)), _mm_castsi128_ps(xmm3)));
                    // vinserti128 ymm2,ymm2,xmm0,1
                    __m256i ymm2 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm2), xmm0, 1);
                    // vinserti128 ymm3,ymm3,xmm1,1
                    __m256i ymm3 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm3), xmm1, 1);
                    // vpunpcklbw ymm0,ymm2,ymm6
                    __m256i ymm0 = _mm256_unpacklo_epi8(ymm2, ymm6);
                    // vpunpcklbw ymm1,ymm3,ymm6
                    __m256i ymm1 = _mm256_unpacklo_epi8(ymm3, ymm6);
                    // vpsadbw xmm2,xmm2,xmm6
                    xmm2 = _mm_sad_epu8(_mm256_castsi256_si128(ymm2), _mm256_castsi256_si128(ymm6));
                    // vpsadbw xmm3,xmm3,xmm6
                    xmm3 = _mm_sad_epu8(_mm256_castsi256_si128(ymm3), _mm256_castsi256_si128(ymm6));
                    // vmovdqa YMMWORD PTR[rdx],ymm0
                    _mm256_store_si256((__m256i*)rdx_ptr, ymm0);
                    // vmovdqa YMMWORD PTR[rdx+rdi*2],ymm1
                    _mm256_store_si256((__m256i*)(rdx_ptr + rdi * 2), ymm1);
                    // vpmaddwd ymm0,ymm0,ymm0
                    ymm0 = _mm256_madd_epi16(ymm0, ymm0);
                    // vpmaddwd ymm1,ymm1,ymm1
                    ymm1 = _mm256_madd_epi16(ymm1, ymm1);
                    // vpaddd xmm4,xmm4,xmm2
                    ymm4 = _mm256_add_epi32(ymm4, _mm256_castsi128_si256(xmm2));
                    // vpaddd ymm5,ymm5,ymm0
                    ymm5 = _mm256_add_epi32(ymm5, ymm0);
                    // vpaddd xmm4,xmm4,xmm3
                    ymm4 = _mm256_add_epi32(ymm4, _mm256_castsi128_si256(xmm3));
                    // vpaddd ymm5,ymm5,ymm1
                    ymm5 = _mm256_add_epi32(ymm5, ymm1);

                    rdx_ptr += r12;
                }
                rax_ptr += rbx * 4;
                rdx_ptr += rdi * 2;
            }
        }

        // vmovhlps xmm1,xmm1,xmm4
        __m128i xmm1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm6)), _mm_castsi128_ps(_mm256_castsi256_si128(ymm4))));
        // vextracti128 xmm2,ymm5,1
        __m128i xmm2 = _mm256_extracti128_si256(ymm5, 1);
        // vpaddd xmm4,xmm4,xmm1
        ymm4 = _mm256_add_epi32(ymm4, _mm256_castsi128_si256(xmm1));
        // vpaddd xmm5,xmm5,xmm2
        ymm5 = _mm256_add_epi32(ymm5, _mm256_castsi128_si256(xmm2));
    }

    // suite0 処理
    // vmovhlps xmm1,xmm1,xmm5
    __m128i xmm1 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm256_castsi256_si128(ymm6)), _mm_castsi128_ps(_mm256_castsi256_si128(ymm5))));
    // vpaddd xmm5,xmm5,xmm1
    ymm5 = _mm256_add_epi32(ymm5, _mm256_castsi128_si256(xmm1));
    // vpshuflw xmm1,xmm5,14
    xmm1 = _mm_shufflelo_epi16(_mm256_castsi256_si128(ymm5), 14);
    // vpaddd xmm5,xmm5,xmm1
    ymm5 = _mm256_add_epi32(ymm5, _mm256_castsi128_si256(xmm1));

    // 浮動小数点変換と計算
    float r7 = 1.0f / (r8 * rdi);
    __m128 xmm4 = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm4));
    __m128 xmm5 = _mm_cvtepi32_ps(_mm256_castsi256_si128(ymm5));
    xmm4 = _mm_mul_ss(xmm4, _mm_set_ss(r7));
    xmm5 = _mm_mul_ss(xmm5, _mm_set_ss(r7));

    // 結果の保存
    _mm_store_ss((float*)mstd, xmm4);
    xmm4 = _mm_mul_ss(xmm4, xmm4);
    xmm5 = _mm_sub_ss(xmm5, xmm4);

    if (_mm_comile_ss(xmm5, _mm_load_ss((float*)&flt_epsilon_sse))) {
        // novarjmp_2
        _mm_store_ss(mstd + 1, _mm256_castps256_ps128(_mm256_castsi256_ps(ymm6)));
        _mm_store_ss(mstd + 2, _mm256_castps256_ps128(_mm256_castsi256_ps(ymm6)));
    } else {
        // vrsqrtss xmm5,xmm5,xmm5
        xmm5 = _mm_rsqrt_ss(xmm5);
        // vrcpss xmm4,xmm4,xmm5
        xmm4 = _mm_rcp_ss(xmm5);
        // vmovss dword ptr[rax+4],xmm4
        _mm_store_ss(mstd + 1, xmm4);
        // vmovss dword ptr[rax+8],xmm5
        _mm_store_ss(mstd + 2, xmm5);
    }

    // finish_4
    _mm_store_ss(mstd + 3, _mm256_castps256_ps128(_mm256_castsi256_ps(ymm6)));

    _mm256_zeroupper();
}

// 元のアセンブラ関数の引数:
// extract_m8_i16_AVX2_16 proc srcp:dword,stride:dword,xdia:dword,ydia:dword,mstd:dword,inputf:dword
// srcp  = rcx
// stride = edx
// xdia  = r8d (画素数, 16bit 単位)
// ydia  = r9d (行数)
// mstd  = [rbp+48]
// inputf= [rbp+56]
extern "C" void extract_m8_i16_AVX2_16(
    const uint8_t* srcp,   // rcx
    int stride,           // edx
    int xdia,             // r8d (画素数, 16bit 単位)
    int ydia,             // r9d (行数)
    float* mstd,          // 集計結果を書き込む (4 個)
    float* inputf         // 出力 (unsigned 16bit のメモリを float* 経由で受け取るが実体は 16bit)
) {
    // レジスタ (変数) の初期化 ---------------------------------------------------
    const uint8_t* rax = srcp;                // 現在行へのポインタ
    const ptrdiff_t rbx = stride;             // ピッチ (バイト)
    int      rdi = xdia;                      // 水平方向ピクセル数 (16bit)
    int      r8d = ydia;                      // 垂直方向ピクセル数
    int16_t* rdx = reinterpret_cast<int16_t*>(inputf);  // 出力先バッファ (実際は 16bit 配列)

    // ループ用定数
    constexpr int r10 = 2;   // y 方向 step (2 行ずつ処理)
    constexpr int r11 = 16;  // x 方向  step (16 word = 32byte) /XMM 用では word 単位
    constexpr int r12 = 32;  // 出力ポインタ増分 (byte)

    const uint8_t* rsi = rax + rbx * 2;       // 2 行下 (偶数行と偶数+2 行をペアに)

    // アキュムレータ (整数: 32bit x8)
    __m256i ymm4 = _mm256_setzero_si256();    // sum
    __m256i ymm5 = _mm256_setzero_si256();    // sumsq
    const __m256i ymm8 = uw_1;                // 16bit 全て 1

    // ----------------------------------------------------------------------------
    // 幅 8 以下 (<=8 pixel) は 128bit パスを使用
    if (rdi <= 8) {
        // アラインメント判定 (16byte)
        const bool aligned = (((uintptr_t)rax & 0x0F) == 0);
        while (r8d > 0) {
            // -------------------- 1 ループで 2 行処理 -------------------------
            // vmovdqa/vmovdqu xmm0, [rax]
            __m128i xmm0 = aligned ? _mm_load_si128((const __m128i*)rax)
                                   : _mm_loadu_si128((const __m128i*)rax);
            // vmovdqa/vmovdqu xmm1, [rsi]
            __m128i xmm1 = aligned ? _mm_load_si128((const __m128i*)rsi)
                                   : _mm_loadu_si128((const __m128i*)rsi);

            // vmovdqa [rdx],xmm0
            _mm_store_si128((__m128i*)rdx, xmm0);
            // vmovdqa [rdx+rdi*2],xmm1  (rdi は word 単位, 2 倍で byte)
            _mm_store_si128((__m128i*)((uint8_t*)rdx + rdi * 2), xmm1);

            // vpmaddwd xmm2,xmm0,ymm8  (pairwise 加算)
            __m128i xmm2 = _mm_madd_epi16(xmm0, _mm256_castsi256_si128(ymm8));
            // vpmaddwd xmm3,xmm1,ymm8
            __m128i xmm3 = _mm_madd_epi16(xmm1, _mm256_castsi256_si128(ymm8));
            // vpmaddwd xmm0,xmm0,xmm0  (square)
            __m128i xmm0sq = _mm_madd_epi16(xmm0, xmm0);
            // vpmaddwd xmm1,xmm1,xmm1
            __m128i xmm1sq = _mm_madd_epi16(xmm1, xmm1);

            // 128bit → 256bit に sign extend してアキュムレータへ加算
            __m256i ymmSum = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm2), xmm3, 1);
            ymm4 = _mm256_add_epi32(ymm4, ymmSum);
            __m256i ymmSq  = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm0sq), xmm1sq, 1);
            ymm5 = _mm256_add_epi32(ymm5, ymmSq);
            
            // ポインタ更新
            rdx = (int16_t*)((uint8_t*)rdx + r11);         // +16byte
            rax += rbx * 4;                                // 次の偶数行ペア
            rsi += rbx * 4;
            rdx = (int16_t*)((uint8_t*)rdx + rdi * 2);     // +width*2
            r8d -= r10;
        }
    } else {
        // ---------------------------------------------------------------------
        // 幅 >8 の場合 256bit パス
        const bool aligned32 = (((uintptr_t)rax & 0x1F) == 0);
        while (r8d > 0) {
            int rcx = 0;  // word 単位オフセット
            while (rcx < rdi) {
                // vmovdqa/vmovdqu ymm0, [rax+2*rcx]
                const uint8_t* ptr0 = rax + (size_t)rcx * 2;
                const uint8_t* ptr1 = rsi + (size_t)rcx * 2;
                __m256i ymm0 = aligned32 ? _mm256_load_si256((const __m256i*)ptr0)
                                          : _mm256_loadu_si256((const __m256i*)ptr0);
                __m256i ymm1 = aligned32 ? _mm256_load_si256((const __m256i*)ptr1)
                                          : _mm256_loadu_si256((const __m256i*)ptr1);

                // vmovdqa [rdx],ymm0
                _mm256_store_si256((__m256i*)rdx, ymm0);
                // vmovdqa [rdx+rdi*2],ymm1
                _mm256_store_si256((__m256i*)((uint8_t*)rdx + rdi * 2), ymm1);

                // vpmaddwd (sum)
                __m256i ymm2 = _mm256_madd_epi16(ymm0, ymm8);
                __m256i ymm3 = _mm256_madd_epi16(ymm1, ymm8);
                // vpmaddwd (sumsq)
                __m256i ymm0sq = _mm256_madd_epi16(ymm0, ymm0);
                __m256i ymm1sq = _mm256_madd_epi16(ymm1, ymm1);

                // accumulate
                ymm4 = _mm256_add_epi32(ymm4, ymm2);
                ymm4 = _mm256_add_epi32(ymm4, ymm3);
                ymm5 = _mm256_add_epi32(ymm5, ymm0sq);
                ymm5 = _mm256_add_epi32(ymm5, ymm1sq);

                rcx += r11;                                   // +16 word
                rdx = (int16_t*)((uint8_t*)rdx + r12);        // +32 byte
            }
            rax += rbx * 4;   // 次の偶数行ペア
            rsi += rbx * 4;
            rdx = (int16_t*)((uint8_t*)rdx + rdi * 2);  // 行分スキップ
            r8d -= r10;                               // 2 行消費
        }
    }

    // ----------------------------------------------------------------------------
    // 水平加算して 32bit sum / sumsq を取り出し ------------------------------
    __m128i lo4 = _mm256_castsi256_si128(ymm4);
    __m128i hi4 = _mm256_extracti128_si256(ymm4, 1);
    __m128i lo5 = _mm256_castsi256_si128(ymm5);
    __m128i hi5 = _mm256_extracti128_si256(ymm5, 1);

    lo4 = _mm_add_epi32(lo4, hi4);   // vpaddd xmm4,xmm4,xmm1
    lo5 = _mm_add_epi32(lo5, hi5);   // vpaddd xmm5,xmm5,xmm2

    // 水平 4 要素 -> 1 要素
    __m128i tmp4 = _mm_shuffle_epi32(lo4, 0x4E);  // swap 高低
    __m128i tmp5 = _mm_shuffle_epi32(lo5, 0x4E);
    lo4 = _mm_add_epi32(lo4, tmp4);
    lo5 = _mm_add_epi32(lo5, tmp5);
    tmp4 = _mm_shuffle_epi32(lo4, 0x11);
    tmp5 = _mm_shuffle_epi32(lo5, 0x11);
    lo4 = _mm_add_epi32(lo4, tmp4);
    lo5 = _mm_add_epi32(lo5, tmp5);

    // 最終スカラー値
    int32_t sum   = _mm_cvtsi128_si32(lo4);
    int32_t sumsq = _mm_cvtsi128_si32(lo5);

    const int nPix = xdia * ydia;            // 総ピクセル数
    const float inv_n = 1.0f / static_cast<float>(nPix);

    float mean  = static_cast<float>(sum)   * inv_n;
    float var   = static_cast<float>(sumsq) * inv_n - mean * mean;

    mstd[0] = mean;              // 平均
    if (var <= FLT_EPSILON) {
        mstd[1] = 0.0f;          // 標準偏差
        mstd[2] = 0.0f;          // 1/標準偏差
    } else {
        float stddev_inv = 1.0f / std::sqrt(var);
        mstd[1] = var * stddev_inv; // sqrt(var) = var * (1/sqrt(var))
        mstd[2] = stddev_inv;
    }
    mstd[3] = 0.0f;              // 互換性のために 0 を格納

    _mm256_zeroupper();
}

extern "C" void extract_m8_i16_AVX2_16_2(
    const uint8_t *srcp, // rcx
    int stride,         // edx
    int xdia,           // r8d
    int ydia,           // r9d 
    float *inputf,      // [rbp+48] (実際は16bit整数配列として使用)
    int32_t *sum,       // [rbp+56]
    int64_t *sumsq      // [rbp+64]
) {
    // レジスタの初期化
    const uint8_t* rax = srcp;
    const ptrdiff_t rbx = stride;
    int rdi = xdia;
    int16_t* rdx = reinterpret_cast<int16_t*>(inputf);
    int r8d = ydia;

    // 定数設定
    const int r10 = 2;  // y方向step (2行ずつ)
    const int r11 = 8;  // x方向step (8 word)
    const int r12 = 16; // 出力ポインタ増分 (byte)

    // 2行目へのポインタ設定
    const uint8_t* rsi = rax + rbx * 2;

    // アキュムレータの初期化
    __m128i xmm4 = _mm_setzero_si128();    // sum (32bit x4)
    __m256i ymm5 = _mm256_setzero_si256(); // sumsq (64bit x4)
    __m128i xmm6 = _mm_setzero_si128();    // ゼロレジスタ
    __m128i xmm7 = _mm_load_si128((__m128i*)&uw_1); // 全bit 1のレジスタ

    // アライメントチェック
    bool aligned = ((uintptr_t)rax & 15) == 0;
    
    // メインループ
    if (aligned) {
        // アライメント済みパス
        while (r8d > 0) {
            int rcx = 0;
            while (rcx < rdi) {
                // vmovdqa xmm0, XMMWORD PTR[rax+2*rcx]
                __m128i xmm0 = _mm_load_si128((__m128i*)(rax + 2 * rcx));
                // vmovdqa xmm1, XMMWORD PTR[rsi+2*rcx]
                __m128i xmm1 = _mm_load_si128((__m128i*)(rsi + 2 * rcx));
                
                // vmovdqa XMMWORD PTR[rdx], xmm0
                _mm_store_si128((__m128i*)rdx, xmm0);
                // vmovdqa XMMWORD PTR[rdx+rdi*2], xmm1
                _mm_store_si128((__m128i*)((char*)rdx + rdi * 2), xmm1);
                
                // vpmaddwd xmm2, xmm0, xmm7
                __m128i xmm2 = _mm_madd_epi16(xmm0, xmm7);
                // vpmaddwd xmm3, xmm1, xmm7
                __m128i xmm3 = _mm_madd_epi16(xmm1, xmm7);
                
                // vpaddd xmm4, xmm4, xmm2
                xmm4 = _mm_add_epi32(xmm4, xmm2);
                // vpaddd xmm4, xmm4, xmm3
                xmm4 = _mm_add_epi32(xmm4, xmm3);
                
                // vmovhlps xmm2, xmm6, xmm0
                xmm2 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(xmm6), _mm_castsi128_ps(xmm0)));
                // vmovhlps xmm3, xmm6, xmm1
                xmm3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(xmm6), _mm_castsi128_ps(xmm1)));
                
                // vinserti128 ymm0, ymm0, xmm2, 1
                __m256i ymm0 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm0), xmm2, 1);
                // vinserti128 ymm1, ymm1, xmm3, 1
                __m256i ymm1 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm1), xmm3, 1);
                
                // vpunpcklwd ymm0, ymm0, ymm6
                ymm0 = _mm256_unpacklo_epi16(ymm0, _mm256_castsi128_si256(xmm6));
                // vpunpcklwd ymm1, ymm1, ymm6
                ymm1 = _mm256_unpacklo_epi16(ymm1, _mm256_castsi128_si256(xmm6));
                
                // vpmulld ymm0, ymm0, ymm0
                ymm0 = _mm256_mullo_epi32(ymm0, ymm0);
                // vpmulld ymm1, ymm1, ymm1
                ymm1 = _mm256_mullo_epi32(ymm1, ymm1);
                
                // vpunpckhdq ymm2, ymm0, ymm6
                __m256i ymm2 = _mm256_unpackhi_epi32(ymm0, _mm256_castsi128_si256(xmm6));
                // vpunpckhdq ymm3, ymm1, ymm6
                __m256i ymm3 = _mm256_unpackhi_epi32(ymm1, _mm256_castsi128_si256(xmm6));
                // vpunpckldq ymm0, ymm0, ymm6
                ymm0 = _mm256_unpacklo_epi32(ymm0, _mm256_castsi128_si256(xmm6));
                // vpunpckldq ymm1, ymm1, ymm6
                ymm1 = _mm256_unpacklo_epi32(ymm1, _mm256_castsi128_si256(xmm6));
                
                // vpaddq ymm0, ymm0, ymm2
                ymm0 = _mm256_add_epi64(ymm0, ymm2);
                // vpaddq ymm1, ymm1, ymm3
                ymm1 = _mm256_add_epi64(ymm1, ymm3);
                
                // vpaddq ymm5, ymm5, ymm0
                ymm5 = _mm256_add_epi64(ymm5, ymm0);
                // vpaddq ymm5, ymm5, ymm1
                ymm5 = _mm256_add_epi64(ymm5, ymm1);
                
                rcx += r11;
                rdx = (int16_t*)((char*)rdx + r12);
            }
            
            rax += rbx * 4;
            rsi += rbx * 4;
            rdx = (int16_t*)((char*)rdx + rdi * 2);
            r8d -= r10;
        }
    } else {
        // 非アライメントパス
        while (r8d > 0) {
            int rcx = 0;
            while (rcx < rdi) {
                // vmovdqu xmm0, XMMWORD PTR[rax+2*rcx]
                __m128i xmm0 = _mm_loadu_si128((__m128i*)(rax + 2 * rcx));
                // vmovdqu xmm1, XMMWORD PTR[rsi+2*rcx]
                __m128i xmm1 = _mm_loadu_si128((__m128i*)(rsi + 2 * rcx));
                
                // vmovdqa XMMWORD PTR[rdx], xmm0
                _mm_store_si128((__m128i*)rdx, xmm0);
                // vmovdqa XMMWORD PTR[rdx+rdi*2], xmm1
                _mm_store_si128((__m128i*)((char*)rdx + rdi * 2), xmm1);
                
                // vpmaddwd xmm2, xmm0, xmm7
                __m128i xmm2 = _mm_madd_epi16(xmm0, xmm7);
                // vpmaddwd xmm3, xmm1, xmm7
                __m128i xmm3 = _mm_madd_epi16(xmm1, xmm7);
                
                // vpaddd xmm4, xmm4, xmm2
                xmm4 = _mm_add_epi32(xmm4, xmm2);
                // vpaddd xmm4, xmm4, xmm3
                xmm4 = _mm_add_epi32(xmm4, xmm3);
                
                // vmovhlps xmm2, xmm6, xmm0
                xmm2 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(xmm6), _mm_castsi128_ps(xmm0)));
                // vmovhlps xmm3, xmm6, xmm1
                xmm3 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(xmm6), _mm_castsi128_ps(xmm1)));
                
                // vinserti128 ymm0, ymm0, xmm2, 1
                __m256i ymm0 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm0), xmm2, 1);
                // vinserti128 ymm1, ymm1, xmm3, 1
                __m256i ymm1 = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm1), xmm3, 1);
                
                // vpunpcklwd ymm0, ymm0, ymm6
                ymm0 = _mm256_unpacklo_epi16(ymm0, _mm256_castsi128_si256(xmm6));
                // vpunpcklwd ymm1, ymm1, ymm6
                ymm1 = _mm256_unpacklo_epi16(ymm1, _mm256_castsi128_si256(xmm6));
                
                // vpmulld ymm0, ymm0, ymm0
                ymm0 = _mm256_mullo_epi32(ymm0, ymm0);
                // vpmulld ymm1, ymm1, ymm1
                ymm1 = _mm256_mullo_epi32(ymm1, ymm1);
                
                // vpunpckhdq ymm2, ymm0, ymm6
                __m256i ymm2 = _mm256_unpackhi_epi32(ymm0, _mm256_castsi128_si256(xmm6));
                // vpunpckhdq ymm3, ymm1, ymm6
                __m256i ymm3 = _mm256_unpackhi_epi32(ymm1, _mm256_castsi128_si256(xmm6));
                // vpunpckldq ymm0, ymm0, ymm6
                ymm0 = _mm256_unpacklo_epi32(ymm0, _mm256_castsi128_si256(xmm6));
                // vpunpckldq ymm1, ymm1, ymm6
                ymm1 = _mm256_unpacklo_epi32(ymm1, _mm256_castsi128_si256(xmm6));
                
                // vpaddq ymm0, ymm0, ymm2
                ymm0 = _mm256_add_epi64(ymm0, ymm2);
                // vpaddq ymm1, ymm1, ymm3
                ymm1 = _mm256_add_epi64(ymm1, ymm3);
                
                // vpaddq ymm5, ymm5, ymm0
                ymm5 = _mm256_add_epi64(ymm5, ymm0);
                // vpaddq ymm5, ymm5, ymm1
                ymm5 = _mm256_add_epi64(ymm5, ymm1);
                
                rcx += r11;
                rdx = (int16_t*)((char*)rdx + r12);
            }
            
            rax += rbx * 4;
            rsi += rbx * 4;
            rdx = (int16_t*)((char*)rdx + rdi * 2);
            r8d -= r10;
        }
    }
    
    // 水平加算
    // vmovhlps xmm0, xmm6, xmm4
    __m128i xmm0 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(xmm6), _mm_castsi128_ps(xmm4)));
    // vextracti128 xmm1, ymm5, 1
    __m128i xmm1 = _mm256_extracti128_si256(ymm5, 1);
    
    // vpaddd xmm4, xmm4, xmm0
    xmm4 = _mm_add_epi32(xmm4, xmm0);
    // vpaddq xmm5, xmm5, xmm1
    __m128i xmm5 = _mm_add_epi64(_mm256_castsi256_si128(ymm5), xmm1);
    
    // vpshufd xmm2, xmm4, 1
    __m128i xmm2 = _mm_shuffle_epi32(xmm4, 1);
    // vmovhlps xmm0, xmm6, xmm5
    xmm0 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(xmm6), _mm_castsi128_ps(xmm5)));
    
    // vpaddd xmm4, xmm4, xmm2
    xmm4 = _mm_add_epi32(xmm4, xmm2);
    // vpaddq xmm5, xmm5, xmm0
    xmm5 = _mm_add_epi64(xmm5, xmm0);
    
    // vmovq qword ptr [rbx], xmm5
    _mm_storel_epi64((__m128i*)sumsq, xmm5);
    // vmovd dword ptr [rax], xmm4
    *sum = _mm_cvtsi128_si32(xmm4);
    
    _mm256_zeroupper();
}

extern "C"
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void extract_m8_FMA3_16(
    const uint8_t* srcp,
    int stride,
    int xdia,
    int ydia,
    float* mstd,
    float* input
) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    __m256 sum = _mm256_setzero_ps();
    __m256 sumsq = _mm256_setzero_ps();

    // Windows ASMと同じく2行を組にし、各xで上段、下段の順に加算する。
    for (int y = 0; y < ydia; y += 2) {
        const uint8_t* src_row0 = srcp + static_cast<ptrdiff_t>(y) * stride * 2;
        const uint8_t* src_row1 = src_row0 + stride * 2;
        float* dst_row0 = input + y * xdia;
        float* dst_row1 = dst_row0 + xdia;
        for (int x = 0; x < xdia; x += 8) {
            const __m128i pixels16_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_row0 + x * 2));
            const __m128i pixels16_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_row1 + x * 2));
            const __m256 pixels0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(pixels16_0));
            const __m256 pixels1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(pixels16_1));
            _mm256_storeu_ps(dst_row0 + x, pixels0);
            _mm256_storeu_ps(dst_row1 + x, pixels1);
            sum = _mm256_add_ps(sum, pixels0);
            sum = _mm256_add_ps(sum, pixels1);
            sumsq = _mm256_fmadd_ps(pixels0, pixels0, sumsq);
            sumsq = _mm256_fmadd_ps(pixels1, pixels1, sumsq);
        }
    }

    const __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    const __m128 sumsq_high = _mm256_extractf128_ps(sumsq, 1);
    __m128 sum_total = _mm_add_ps(_mm256_castps256_ps128(sum), sum_high);
    __m128 sumsq_total = _mm_add_ps(_mm256_castps256_ps128(sumsq), sumsq_high);
    sum_total = _mm_add_ps(sum_total, _mm_movehl_ps(sum_high, sum_total));
    sumsq_total = _mm_add_ps(sumsq_total, _mm_movehl_ps(sumsq_high, sumsq_total));
    sum_total = _mm_add_ss(sum_total, _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(sum_total), 14)));
    sumsq_total = _mm_add_ss(sumsq_total, _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(sumsq_total), 14)));

    const __m128 inv_count = _mm_rcp_ss(_mm_set_ss(static_cast<float>(xdia * ydia)));
    const __m128 mean = _mm_mul_ss(sum_total, inv_count);
    const __m128 average_square = _mm_mul_ss(sumsq_total, inv_count);
    const __m128 variance = _mm_sub_ss(average_square, _mm_mul_ss(mean, mean));

    mstd[0] = _mm_cvtss_f32(mean);
    if (_mm_cvtss_f32(variance) <= FLT_EPSILON) {
        mstd[1] = 0.0f;
        mstd[2] = 0.0f;
    } else {
        const __m128 inv_stddev = _mm_rsqrt_ss(variance);
        mstd[1] = _mm_cvtss_f32(_mm_rcp_ss(inv_stddev));
        mstd[2] = _mm_cvtss_f32(inv_stddev);
    }
    mstd[3] = 0.0f;

    _mm256_zeroupper();
}

extern "C"
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void extract_m8_FMA3_32(
    const uint8_t* srcp,
    int stride,
    int xdia,
    int ydia,
    float* mstd,
    float* input
) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    __m256 sum = _mm256_setzero_ps();
    __m256 sumsq = _mm256_setzero_ps();

    for (int y = 0; y < ydia; y += 2) {
        const uint8_t* src_row0 = srcp + static_cast<ptrdiff_t>(y) * stride * 2;
        const uint8_t* src_row1 = src_row0 + stride * 2;
        float* dst_row0 = input + y * xdia;
        float* dst_row1 = dst_row0 + xdia;
        for (int x = 0; x < xdia; x += 8) {
            const __m256 pixels0 = _mm256_loadu_ps(reinterpret_cast<const float*>(src_row0) + x);
            const __m256 pixels1 = _mm256_loadu_ps(reinterpret_cast<const float*>(src_row1) + x);
            _mm256_storeu_ps(dst_row0 + x, pixels0);
            _mm256_storeu_ps(dst_row1 + x, pixels1);
            sum = _mm256_add_ps(sum, pixels0);
            sum = _mm256_add_ps(sum, pixels1);
            sumsq = _mm256_fmadd_ps(pixels0, pixels0, sumsq);
            sumsq = _mm256_fmadd_ps(pixels1, pixels1, sumsq);
        }
    }

    const __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    const __m128 sumsq_high = _mm256_extractf128_ps(sumsq, 1);
    __m128 sum_total = _mm_add_ps(_mm256_castps256_ps128(sum), sum_high);
    __m128 sumsq_total = _mm_add_ps(_mm256_castps256_ps128(sumsq), sumsq_high);
    sum_total = _mm_add_ps(sum_total, _mm_movehl_ps(sum_high, sum_total));
    sumsq_total = _mm_add_ps(sumsq_total, _mm_movehl_ps(sumsq_high, sumsq_total));
    sum_total = _mm_add_ss(sum_total, _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(sum_total), 14)));
    sumsq_total = _mm_add_ss(sumsq_total, _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(sumsq_total), 14)));

    const __m128 inv_count = _mm_rcp_ss(_mm_set_ss(static_cast<float>(xdia * ydia)));
    const __m128 mean = _mm_mul_ss(sum_total, inv_count);
    const __m128 average_square = _mm_mul_ss(sumsq_total, inv_count);
    const __m128 variance = _mm_sub_ss(average_square, _mm_mul_ss(mean, mean));

    mstd[0] = _mm_cvtss_f32(mean);
    if (_mm_cvtss_f32(variance) <= FLT_EPSILON) {
        mstd[1] = 0.0f;
        mstd[2] = 0.0f;
    } else {
        const __m128 inv_stddev = _mm_rsqrt_ss(variance);
        mstd[1] = _mm_cvtss_f32(_mm_rcp_ss(inv_stddev));
        mstd[2] = _mm_cvtss_f32(inv_stddev);
    }
    mstd[3] = 0.0f;

    _mm256_zeroupper();
}

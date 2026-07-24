#include "nnedi3_intrinsic_AVX512.h"

extern "C" std::uint32_t nnedi3_avx512_build_marker() noexcept
{
    // 非対応CPUでも安全に参照できるよう、このmarkerにはAVX-512命令を含めない。
    return UINT32_C(0x41565835);
}

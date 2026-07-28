#pragma once

#include <cstddef>
#include <cstdint>

namespace nnedi3_backend {

constexpr int CPU_SSE2 = 0x00000020;
constexpr int CPU_SSE41 = 0x00000400;
constexpr int CPU_AVX = 0x00000800;
constexpr int CPU_AVX2 = 0x00002000;
constexpr int CPU_FMA3 = 0x00004000;
constexpr int CPU_FMA4 = 0x00080000;
constexpr int CPU_AVX512F = 0x00100000;
constexpr int CPU_AVX512DQ = 0x00200000;
constexpr int CPU_AVX512BW = 0x02000000;
constexpr int CPU_AVX512VL = 0x04000000;
constexpr int CPU_AVXVNNI = 0x20000000;
constexpr int CPU_AVX512VNNI = 0x40000000;

constexpr int AVX512_REQUIRED = CPU_AVX2 | CPU_FMA3 | CPU_AVX512F
    | CPU_AVX512DQ | CPU_AVX512BW | CPU_AVX512VL;
constexpr int AVXVNNI_REQUIRED = CPU_AVX2 | CPU_FMA3 | CPU_AVXVNNI;
constexpr int AVX512VNNI_REQUIRED = AVX512_REQUIRED | CPU_AVX512VNNI;

enum class Platform {
    Windows,
    Linux,
};

enum class Backend {
    C,
    SSE2,
    SSE41,
    AVX,
    AVX2,
    AVX2FMA3,
    AVX2FMA3VNNI,
    AVX2FMA4,
    AVX512,
    AVX512VNNI,
};

enum class WeightLayout {
    NeuronMajor,
    LegacySIMD,
    AVX2,
    AVX512,
    AVX512Prescreener,
};

enum class SelectionError {
    None,
    InvalidOpt,
    MissingFeatures,
};

enum class PredictorDot {
    Existing,
    CInt16,
    AVXVNNIInt16,
    AVX512Float,
    AVX512Int16,
    AVX512VNNIInt16,
};

struct BackendSelection {
    Backend backend;
    int normalized_opt;
    int missing_features;
    SelectionError error;
};

struct KernelSet {
    Backend requested_backend;
    Backend kernel_backend;
    WeightLayout prescreener_weights;
    WeightLayout predictor_weights;
    bool has_sse2;
    bool has_sse41;
    bool has_avx;
    bool has_avx2;
    bool has_fma3;
};

struct PredictorPlan {
    PredictorDot dot;
    WeightLayout layout;
};

constexpr bool has_all_features(const int cpu_flags, const int required) {
    return (cpu_flags & required) == required;
}

constexpr bool is_avx512_backend(const Backend backend) {
    return backend == Backend::AVX512 || backend == Backend::AVX512VNNI;
}

constexpr Backend backend_from_legacy_opt(const int opt) {
    return opt == 1 ? Backend::C
        : opt == 2 ? Backend::SSE2
        : opt == 3 ? Backend::SSE41
        : opt == 4 ? Backend::AVX
        : opt == 5 ? Backend::AVX2
        : opt == 6 ? Backend::AVX2FMA3
        : opt == 7 ? Backend::AVX2FMA4
        : opt == 8 ? Backend::AVX2FMA3VNNI
        : opt == 9 ? Backend::AVX512
        : opt == 10 ? Backend::AVX512VNNI
        : Backend::C;
}

constexpr BackendSelection select_windows_backend(const int requested_opt, const int cpu_flags) {
    if (requested_opt < 0 || requested_opt > 10) {
        return {Backend::C, requested_opt, 0, SelectionError::InvalidOpt};
    }
    if (requested_opt == 10) {
        const int missing = AVX512VNNI_REQUIRED & ~cpu_flags;
        return {Backend::AVX512VNNI, 10, missing,
            missing == 0 ? SelectionError::None : SelectionError::MissingFeatures};
    }
    if (requested_opt == 9) {
        const int missing = AVX512_REQUIRED & ~cpu_flags;
        return {Backend::AVX512, 9, missing,
            missing == 0 ? SelectionError::None : SelectionError::MissingFeatures};
    }
    if (requested_opt == 8) {
        const int missing = AVXVNNI_REQUIRED & ~cpu_flags;
        return {Backend::AVX2FMA3VNNI, 8, missing,
            missing == 0 ? SelectionError::None : SelectionError::MissingFeatures};
    }
    if (requested_opt >= 5) {
        const int required = CPU_AVX2 | CPU_FMA3;
        const int missing = required & ~cpu_flags;
        return {Backend::AVX2FMA3, 6, missing,
            missing == 0 ? SelectionError::None : SelectionError::MissingFeatures};
    }
    if (requested_opt != 0) {
        return {backend_from_legacy_opt(requested_opt), requested_opt, 0, SelectionError::None};
    }

    if (has_all_features(cpu_flags, AVXVNNI_REQUIRED)) {
        return {Backend::AVX2FMA3VNNI, 8, 0, SelectionError::None};
    }
    if (has_all_features(cpu_flags, CPU_AVX2 | CPU_FMA3)) {
        return {Backend::AVX2FMA3, 6, 0, SelectionError::None};
    }
    if ((cpu_flags & CPU_AVX) != 0) {
        return {Backend::AVX, 4, 0, SelectionError::None};
    }
    if ((cpu_flags & CPU_SSE41) != 0) {
        return {Backend::SSE41, 3, 0, SelectionError::None};
    }
    if ((cpu_flags & CPU_SSE2) != 0) {
        return {Backend::SSE2, 2, 0, SelectionError::None};
    }
    return {Backend::C, 1, 0, SelectionError::None};
}

constexpr BackendSelection select_linux_backend(const int requested_opt, const int cpu_flags) {
    if (requested_opt < 0 || requested_opt > 10) {
        return {Backend::C, requested_opt, 0, SelectionError::InvalidOpt};
    }
    if (requested_opt == 10) {
        const int missing = AVX512VNNI_REQUIRED & ~cpu_flags;
        return {Backend::AVX512VNNI, 10, missing,
            missing == 0 ? SelectionError::None : SelectionError::MissingFeatures};
    }
    if (requested_opt == 9) {
        const int missing = AVX512_REQUIRED & ~cpu_flags;
        return {Backend::AVX512, 9, missing,
            missing == 0 ? SelectionError::None : SelectionError::MissingFeatures};
    }
    if (requested_opt == 8) {
        const int missing = AVXVNNI_REQUIRED & ~cpu_flags;
        return {Backend::AVX2FMA3VNNI, 8, missing,
            missing == 0 ? SelectionError::None : SelectionError::MissingFeatures};
    }

    const bool has_avx2_fma3 = has_all_features(cpu_flags, CPU_AVX2 | CPU_FMA3);
    if (requested_opt == 0 && has_all_features(cpu_flags, AVXVNNI_REQUIRED)) {
        return {Backend::AVX2FMA3VNNI, 8, 0, SelectionError::None};
    }
    if (requested_opt == 0 || requested_opt == 5 || requested_opt == 6 || requested_opt == 7) {
        return has_avx2_fma3
            ? BackendSelection{Backend::AVX2FMA3, 6, 0, SelectionError::None}
            : BackendSelection{Backend::C, 1, 0, SelectionError::None};
    }
    return {Backend::C, 1, 0, SelectionError::None};
}

constexpr BackendSelection select_backend(const Platform platform,
    const int requested_opt, const int cpu_flags) {
    return platform == Platform::Windows
        ? select_windows_backend(requested_opt, cpu_flags)
        : select_linux_backend(requested_opt, cpu_flags);
}

constexpr KernelSet make_kernel_set(const Backend backend) {
    const bool is_avx512 = is_avx512_backend(backend);
    const Backend kernel_backend = is_avx512 ? Backend::AVX2FMA3 : backend;
    const bool has_sse2 = kernel_backend != Backend::C;
    const bool has_sse41 = kernel_backend == Backend::SSE41 || kernel_backend == Backend::AVX
        || kernel_backend == Backend::AVX2 || kernel_backend == Backend::AVX2FMA3
        || kernel_backend == Backend::AVX2FMA3VNNI
        || kernel_backend == Backend::AVX2FMA4;
    const bool has_avx = kernel_backend == Backend::AVX || kernel_backend == Backend::AVX2
        || kernel_backend == Backend::AVX2FMA3 || kernel_backend == Backend::AVX2FMA3VNNI
        || kernel_backend == Backend::AVX2FMA4;
    const bool has_avx2 = kernel_backend == Backend::AVX2 || kernel_backend == Backend::AVX2FMA3
        || kernel_backend == Backend::AVX2FMA3VNNI || kernel_backend == Backend::AVX2FMA4;
    const bool has_fma3 = kernel_backend == Backend::AVX2FMA3
        || kernel_backend == Backend::AVX2FMA3VNNI;
    const WeightLayout layout = has_avx2 ? WeightLayout::AVX2
        : has_sse2 ? WeightLayout::LegacySIMD : WeightLayout::NeuronMajor;
    const WeightLayout prescreenerLayout = is_avx512
        ? WeightLayout::AVX512Prescreener : layout;
    return {backend, kernel_backend, prescreenerLayout, layout, has_sse2, has_sse41,
        has_avx, has_avx2, has_fma3};
}

constexpr KernelSet kernel_set_from_opt(const int normalized_opt) {
    return make_kernel_set(backend_from_legacy_opt(normalized_opt));
}

constexpr PredictorPlan make_predictor_plan(const KernelSet& kernels,
    const bool int16_predictor, const int bits_per_pixel) {
    if (int16_predictor && bits_per_pixel > 14) {
        return {PredictorDot::CInt16, WeightLayout::NeuronMajor};
    }
    if (kernels.requested_backend == Backend::AVX512
        || kernels.requested_backend == Backend::AVX512VNNI) {
        return {int16_predictor
                ? (kernels.requested_backend == Backend::AVX512VNNI
                    ? PredictorDot::AVX512VNNIInt16 : PredictorDot::AVX512Int16)
                : PredictorDot::AVX512Float,
            WeightLayout::AVX512};
    }
    if (kernels.requested_backend == Backend::AVX2FMA3VNNI && int16_predictor)
        return {PredictorDot::AVXVNNIInt16, WeightLayout::AVX2};
    return {PredictorDot::Existing, kernels.predictor_weights};
}

constexpr std::size_t predictor_matrix_index(const WeightLayout layout,
    const bool int16_predictor, const int neuron, const int sample, const int asize) {
    const std::size_t group_base = static_cast<std::size_t>((neuron >> 2) << 2) * asize;
    const int neuron_in_group = neuron & 3;
    if (layout == WeightLayout::NeuronMajor) {
        return static_cast<std::size_t>(neuron) * asize + sample;
    }
    if (layout == WeightLayout::LegacySIMD) {
        const int chunk = int16_predictor ? 8 : 4;
        return group_base + static_cast<std::size_t>(sample / chunk) * (chunk * 4)
            + neuron_in_group * chunk + sample % chunk;
    }
    if (layout == WeightLayout::AVX2) {
        const int chunk = int16_predictor ? 16 : 8;
        return group_base + static_cast<std::size_t>(sample / chunk) * (chunk * 4)
            + neuron_in_group * chunk + sample % chunk;
    }
    if (!int16_predictor) {
        return group_base + static_cast<std::size_t>(sample >> 4) * 64
            + (neuron_in_group << 4) + (sample & 15);
    }
    if (asize == 48) {
        return sample < 32
            ? group_base + (neuron_in_group << 5) + sample
            : group_base + 128 + (neuron_in_group << 4) + (sample - 32);
    }
    return group_base + static_cast<std::size_t>(sample >> 5) * 128
        + (neuron_in_group << 5) + (sample & 31);
}

constexpr bool uses_simd_layout(const WeightLayout layout) {
    return layout != WeightLayout::NeuronMajor;
}

constexpr bool uses_avx2_layout(const WeightLayout layout) {
    return layout == WeightLayout::AVX2;
}

constexpr bool uses_avx512_layout(const WeightLayout layout) {
    return layout == WeightLayout::AVX512;
}

constexpr bool uses_avx512_prescreener_layout(const WeightLayout layout) {
    return layout == WeightLayout::AVX512Prescreener;
}

}

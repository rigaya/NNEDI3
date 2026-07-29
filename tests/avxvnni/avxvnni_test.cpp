#include "nnedi3_intrinsic_AVXVNNI.h"
#include "nnedi3_intrinsic_AVX512.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

extern "C" {

void computeNetwork0_i16_AVX2(
    const float* input, const float* weights, std::uint8_t* result);
void computeNetwork0new_AVX2(
    const float* input, const float* weights, std::uint8_t* result);
void dotProd_m32_m16_i16_AVX2(const float* data, const float* weights,
    float* vals, int n, int len, const float* istd);
void dotProd_m48_m16_i16_AVX2(const float* data, const float* weights,
    float* vals, int n, int len, const float* istd);

}

namespace {

constexpr int skippedExitCode = 77;
volatile float benchmarkSink = 0.0f;

struct CpuFeatures {
    bool avxState;
    bool avx512State;
    bool fma3;
    bool avx2;
    bool avxVnni;
    bool avx512;
    bool avx512Vnni;
};

void cpuid(int registers[4], const int leaf, const int subleaf)
{
#if defined(_MSC_VER)
    __cpuidex(registers, leaf, subleaf);
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __cpuid_count(static_cast<unsigned int>(leaf),
        static_cast<unsigned int>(subleaf), eax, ebx, ecx, edx);
    registers[0] = static_cast<int>(eax);
    registers[1] = static_cast<int>(ebx);
    registers[2] = static_cast<int>(ecx);
    registers[3] = static_cast<int>(edx);
#endif
}

std::uint64_t xgetbv0()
{
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned int eax = 0;
    unsigned int edx = 0;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

CpuFeatures detectCpuFeatures()
{
    int registers[4]{};
    cpuid(registers, 0, 0);
    const int maximumLeaf = registers[0];
    if (maximumLeaf < 1) return {};

    cpuid(registers, 1, 0);
    const bool osxsave = (registers[2] & (1 << 27)) != 0;
    const bool avx = (registers[2] & (1 << 28)) != 0;
    const bool fma3 = (registers[2] & (1 << 12)) != 0;
    const bool avxState = osxsave && avx && (xgetbv0() & 0x06u) == 0x06u;
    const std::uint64_t xcr0 = osxsave ? xgetbv0() : 0;
    const bool avx512State = avxState && (xcr0 & 0xe0u) == 0xe0u;
    if (maximumLeaf < 7)
        return {avxState, avx512State, fma3, false, false, false, false};

    cpuid(registers, 7, 0);
    const int maximumSubleaf = registers[0];
    const bool avx2 = (registers[1] & (1 << 5)) != 0;
    const int avx512Mask = (1 << 16) | (1 << 17) | (1 << 30)
        | static_cast<int>(1u << 31);
    const bool avx512 = (registers[1] & avx512Mask) == avx512Mask;
    const bool avx512Vnni = (registers[2] & (1 << 11)) != 0;
    bool avxVnni = false;
    if (maximumSubleaf >= 1) {
        cpuid(registers, 7, 1);
        avxVnni = (registers[0] & (1 << 4)) != 0;
    }
    return {avxState, avx512State, fma3, avx2, avxVnni,
        avx512, avx512Vnni};
}

template<typename T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(const std::size_t count)
        : storage_(count * sizeof(T) + 63)
    {
        const auto address = reinterpret_cast<std::uintptr_t>(storage_.data());
        data_ = reinterpret_cast<T*>((address + 63u) & ~std::uintptr_t(63u));
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    std::vector<std::uint8_t> storage_;
    T* data_ = nullptr;
};

using DotProduct = void (*)(const float*, const float*, float*, int, int,
    const float*);

DotProduct avx2DotProduct(const int length)
{
    return length == 48
        ? dotProd_m48_m16_i16_AVX2 : dotProd_m32_m16_i16_AVX2;
}

DotProduct avxVnniDotProduct(const int length)
{
    return length == 48
        ? dotProd_m48_m16_i16_AVXVNNI : dotProd_m32_m16_i16_AVXVNNI;
}

DotProduct avx512DotProduct(const int length)
{
    return length == 48
        ? dotProd_m48_m16_i16_AVX512 : dotProd_m32_m16_i16_AVX512;
}

DotProduct avx512VnniDotProduct(const int length)
{
    return length == 48
        ? dotProd_m48_m16_i16_AVX512VNNI
        : dotProd_m32_m16_i16_AVX512VNNI;
}

bool sameFloats(const float* left, const float* right, const int count)
{
    return std::memcmp(left, right,
        static_cast<std::size_t>(count) * sizeof(float)) == 0;
}

void fillIntegers(std::int16_t* destination, const std::size_t count,
    std::mt19937& random)
{
    std::uniform_int_distribution<int> distribution(-30000, 30000);
    for (std::size_t i = 0; i < count; ++i)
        destination[i] = static_cast<std::int16_t>(distribution(random));
}

void fillFloats(float* destination, const std::size_t count,
    std::mt19937& random)
{
    std::uniform_real_distribution<float> distribution(-0.01f, 0.01f);
    for (std::size_t i = 0; i < count; ++i)
        destination[i] = distribution(random);
}

bool testPredictor(std::mt19937& random)
{
    constexpr int trials = 50;
    for (const int length : {32, 48, 64, 96, 128, 192, 288}) {
        for (const int neurons : {16, 32, 64, 128}) {
            AlignedBuffer<std::int16_t> input(length);
            AlignedBuffer<std::int16_t> weights(
                static_cast<std::size_t>(neurons) * length
                + static_cast<std::size_t>(neurons) * sizeof(float));
            AlignedBuffer<float> avx2(neurons);
            AlignedBuffer<float> avxVnni(neurons);
            AlignedBuffer<float> avx512(neurons);
            AlignedBuffer<float> avx512Vnni(neurons);
            float* const scaleBias = reinterpret_cast<float*>(
                weights.data() + static_cast<std::size_t>(neurons) * length);
            const float inverseStdDev = 0.003f;

            for (int trial = 0; trial < trials; ++trial) {
                fillIntegers(input.data(), length, random);
                fillIntegers(weights.data(),
                    static_cast<std::size_t>(neurons) * length, random);
                fillFloats(scaleBias, static_cast<std::size_t>(neurons) * 2,
                    random);
                avx2DotProduct(length)(reinterpret_cast<float*>(input.data()),
                    reinterpret_cast<float*>(weights.data()), avx2.data(),
                    neurons, length, &inverseStdDev);
                avxVnniDotProduct(length)(
                    reinterpret_cast<float*>(input.data()),
                    reinterpret_cast<float*>(weights.data()), avxVnni.data(),
                    neurons, length, &inverseStdDev);
                avx512DotProduct(length)(reinterpret_cast<float*>(input.data()),
                    reinterpret_cast<float*>(weights.data()), avx512.data(),
                    neurons, length, &inverseStdDev);
                avx512VnniDotProduct(length)(
                    reinterpret_cast<float*>(input.data()),
                    reinterpret_cast<float*>(weights.data()), avx512Vnni.data(),
                    neurons, length, &inverseStdDev);
                if (!sameFloats(avx2.data(), avxVnni.data(), neurons)) {
                    std::fprintf(stderr,
                        "予測器が不一致です: len=%d, n=%d, trial=%d\n",
                        length, neurons, trial);
                    return false;
                }
                if (!sameFloats(avx512.data(), avx512Vnni.data(), neurons)) {
                    std::fprintf(stderr,
                        "AVX512予測器が不一致です: len=%d, n=%d, trial=%d\n",
                        length, neurons, trial);
                    return false;
                }
            }
        }
    }
    std::puts("予測器: AVX2とAVX-VNNI、AVX512とAVX512-VNNIの出力がbit単位で一致しました");
    return true;
}

bool testPrescreener(std::mt19937& random)
{
    constexpr int trials = 10000;
    AlignedBuffer<std::int16_t> oldInput(48);
    AlignedBuffer<std::int16_t> oldWeights(192 + 64 * 2);
    AlignedBuffer<std::int16_t> newInput(64);
    AlignedBuffer<std::int16_t> newWeights(256 + 32 * 2);
    float* const oldFloatWeights = reinterpret_cast<float*>(
        oldWeights.data() + 192);
    float* const newFloatWeights = reinterpret_cast<float*>(
        newWeights.data() + 256);

    for (int trial = 0; trial < trials; ++trial) {
        fillIntegers(oldInput.data(), 48, random);
        fillIntegers(oldWeights.data(), 192, random);
        fillFloats(oldFloatWeights, 64, random);
        fillIntegers(newInput.data(), 64, random);
        fillIntegers(newWeights.data(), 256, random);
        fillFloats(newFloatWeights, 32, random);

        std::uint8_t avx2Old = 0;
        std::uint8_t avxVnniOld = 0;
        computeNetwork0_i16_AVX2(reinterpret_cast<float*>(oldInput.data()),
            reinterpret_cast<float*>(oldWeights.data()), &avx2Old);
        computeNetwork0_i16_AVXVNNI(
            reinterpret_cast<float*>(oldInput.data()),
            reinterpret_cast<float*>(oldWeights.data()), &avxVnniOld);
        if (avx2Old != avxVnniOld) {
            std::fprintf(stderr,
                "旧prescreenerが不一致です: trial=%d\n", trial);
            return false;
        }

        std::uint8_t avx2New[4]{};
        std::uint8_t avxVnniNew[4]{};
        computeNetwork0new_AVX2(reinterpret_cast<float*>(newInput.data()),
            reinterpret_cast<float*>(newWeights.data()), avx2New);
        computeNetwork0new_AVXVNNI(
            reinterpret_cast<float*>(newInput.data()),
            reinterpret_cast<float*>(newWeights.data()), avxVnniNew);
        if (std::memcmp(avx2New, avxVnniNew, sizeof(avx2New)) != 0) {
            std::fprintf(stderr,
                "新prescreenerが不一致です: trial=%d\n", trial);
            return false;
        }
    }
    std::puts("prescreener: AVX2とAVX-VNNIの出力がbit単位で一致しました");
    return true;
}

double measurePredictor(const DotProduct function, const int iterations,
    const float* input, const float* weights, float* output,
    const int neurons, const int length, const float* inverseStdDev)
{
    const auto start = std::chrono::steady_clock::now();
    float checksum = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        function(input, weights, output, neurons, length, inverseStdDev);
        checksum += output[i & (neurons - 1)];
    }
    const auto stop = std::chrono::steady_clock::now();
    benchmarkSink += checksum;
    const double nanoseconds = std::chrono::duration<double, std::nano>(
        stop - start).count();
    return nanoseconds / iterations;
}

void runBenchmark(const int iterations, std::mt19937& random)
{
    constexpr int neurons = 64;
    constexpr int rounds = 9;
    std::printf("4経路予測器ベンチマーク: n=%d, %d回 x %dラウンド（中央値）\n",
        neurons, iterations, rounds);
    std::puts("  len   AVX2+FMA3   AVX-VNNI      AVX512  AVX512-VNNI");
    for (const int length : {32, 48, 64, 96, 128, 192, 288}) {
        AlignedBuffer<std::int16_t> input(length);
        AlignedBuffer<std::int16_t> weights(
            static_cast<std::size_t>(neurons) * length
            + static_cast<std::size_t>(neurons) * sizeof(float));
        AlignedBuffer<float> output(neurons);
        float* const scaleBias = reinterpret_cast<float*>(
            weights.data() + static_cast<std::size_t>(neurons) * length);
        const float inverseStdDev = 0.003f;
        fillIntegers(input.data(), length, random);
        fillIntegers(weights.data(), static_cast<std::size_t>(neurons) * length,
            random);
        fillFloats(scaleBias, static_cast<std::size_t>(neurons) * 2, random);

        const DotProduct functions[] = {avx2DotProduct(length),
            avxVnniDotProduct(length), avx512DotProduct(length),
            avx512VnniDotProduct(length)};
        for (int i = 0; i < 200; ++i) {
            for (const DotProduct function : functions)
                function(reinterpret_cast<float*>(input.data()),
                    reinterpret_cast<float*>(weights.data()), output.data(),
                    neurons, length, &inverseStdDev);
        }

        std::vector<double> times[4];
        for (int round = 0; round < rounds; ++round) {
            for (int order = 0; order < 4; ++order) {
                const int index = (round & 1) == 0 ? order : 3 - order;
                times[index].push_back(measurePredictor(functions[index],
                    iterations, reinterpret_cast<float*>(input.data()),
                    reinterpret_cast<float*>(weights.data()), output.data(),
                    neurons, length, &inverseStdDev));
            }
        }
        double medians[4]{};
        for (int i = 0; i < 4; ++i) {
            std::sort(times[i].begin(), times[i].end());
            medians[i] = times[i][rounds / 2];
        }
        std::printf("  %3d %10.2f %10.2f %11.2f %13.2f ns/call\n",
            length, medians[0], medians[1], medians[2], medians[3]);
    }
}

} // 無名名前空間

int main(const int argc, char** argv)
{
    bool runCorrectness = argc == 1;
    bool runPerformance = argc == 1;
    int iterations = 20000;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--correctness") {
            runCorrectness = true;
        } else if (argument == "--benchmark") {
            runPerformance = true;
        } else if (argument == "--iterations" && i + 1 < argc) {
            iterations = std::max(1, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr,
                "使い方: %s [--correctness] [--benchmark] [--iterations 回数]\n",
                argv[0]);
            return 2;
        }
    }

    const CpuFeatures features = detectCpuFeatures();
    std::printf("CPU機能: AVX状態=%s, AVX2=%s, FMA3=%s, AVX-VNNI=%s, "
        "AVX512状態=%s, AVX512=%s, AVX512-VNNI=%s\n",
        features.avxState ? "yes" : "no", features.avx2 ? "yes" : "no",
        features.fma3 ? "yes" : "no", features.avxVnni ? "yes" : "no",
        features.avx512State ? "yes" : "no", features.avx512 ? "yes" : "no",
        features.avx512Vnni ? "yes" : "no");
    if (!features.avxState || !features.avx2 || !features.fma3
        || !features.avxVnni || !features.avx512State || !features.avx512
        || !features.avx512Vnni) {
        std::puts("4経路比較の実行条件を満たさないためスキップします");
        return skippedExitCode;
    }

    std::mt19937 random(0x41565856u);
    if (runCorrectness
        && (!testPredictor(random) || !testPrescreener(random)))
        return 1;
    if (runPerformance) runBenchmark(iterations, random);
    return 0;
}

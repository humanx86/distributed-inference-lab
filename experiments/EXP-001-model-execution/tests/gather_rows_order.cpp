#include "gather_rows_cpu.hpp"
#include "gather_rows_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t warmup = 5;
constexpr std::size_t measured = 20;
constexpr std::uint32_t data_seed = 17;
constexpr std::uint32_t order_seed = 42;
constexpr std::uint64_t small_bytes = 64 * 1024;
constexpr std::uint64_t alignment = 32 * sizeof(float);
static_assert(sizeof(float) == 4);

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::size_t element_count(bool large, int l2_bytes) {
    const std::uint64_t target = large ? std::uint64_t{2} * static_cast<std::uint64_t>(l2_bytes)
                                       : small_bytes;
    const std::uint64_t bytes = ((target + alignment - 1) / alignment) * alignment;

    if (bytes / sizeof(float) > std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t)) {
        throw std::length_error("Planned index vector exceeds addressable size");
    }
    return static_cast<std::size_t>(bytes / sizeof(float));
}

void print_plan(int l2_bytes) {
    std::cout << "size,rows,cols,input_bytes,indices_bytes,output_bytes\n";
    for (const bool large : {false, true}) {
        const auto elements = element_count(large, l2_bytes);
        for (const std::size_t cols : {1u, 32u}) {
            const auto rows = elements / cols;
            std::cout << (large ? "large" : "small") << ',' << rows << ',' << cols
                      << ',' << elements * sizeof(float) << ',' << rows * sizeof(std::int64_t)
                      << ',' << elements * sizeof(float) << '\n';
        }
    }
}

void run_pair(const std::string& size_name, std::size_t cols,
              bool shuffled_first, int l2_bytes) {
    const auto elements = element_count(size_name == "large", l2_bytes);
    const auto rows = elements / cols;
    Matrix input{rows, cols, std::vector<float>(elements)};

    std::mt19937 data_rng(data_seed);
    for (float& value : input.data) {
        value = static_cast<float>(data_rng() >> 8);
    }

    std::vector<std::int64_t> sequential(rows);
    std::iota(sequential.begin(), sequential.end(), std::int64_t{0});
    auto shuffled = sequential;
    std::mt19937 order_rng(order_seed);
    std::shuffle(shuffled.begin(), shuffled.end(), order_rng);

    struct Samples {
        std::vector<float> device;
        std::vector<GatherHostSample> host;
    };
    const std::array<bool, 2> orders{shuffled_first, !shuffled_first};
    std::array<Samples, 2> arms;
    for (std::size_t arm = 0; arm < arms.size(); ++arm) {
        const auto& indices = orders[arm] ? shuffled : sequential;
        auto& samples = arms[arm];
        samples.device.resize(warmup + measured);
        samples.host.resize(warmup + measured);
        const auto actual = gather_rows_cuda(input, indices, samples.device, samples.host);
        const auto expected = gather_rows_cpu(input, indices);
        if (actual.rows != rows || actual.cols != cols || actual.data != expected.data) {
            throw std::runtime_error(orders[arm] ? "Shuffled gather disagrees with CPU"
                                                : "Sequential gather disagrees with CPU");
        }
        for (std::size_t i = 0; i < samples.device.size(); ++i) {
            const auto& h = samples.host[i];
            if (!std::isfinite(samples.device[i]) || samples.device[i] < 0 ||
                !std::isfinite(h.enqueue_us) || !std::isfinite(h.synchronized_us) ||
                h.enqueue_us < 0 || h.synchronized_us < h.enqueue_us) {
                throw std::runtime_error("Invalid timing sample");
            }
        }
    }

    std::cout << "size,rows,cols,arm_position,order,iteration,phase,"
                 "device_event_us,host_enqueue_us,host_synchronized_us\n"
              << std::fixed << std::setprecision(6);
    for (std::size_t arm = 0; arm < arms.size(); ++arm) {
        for (std::size_t i = 0; i < warmup + measured; ++i) {
            const auto& samples = arms[arm];
            std::cout << size_name << ',' << rows << ',' << cols << ',' << arm + 1 << ','
                      << (orders[arm] ? "shuffled" : "sequential") << ',' << i << ','
                      << (i < warmup ? "warmup" : "measured") << ',' << samples.device[i]
                      << ',' << samples.host[i].enqueue_us << ',' << samples.host[i].synchronized_us
                      << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool plan = argc == 2 && std::string(argv[1]) == "--plan";
        if (!plan && (argc != 4 ||
                      (std::string(argv[1]) != "small" && std::string(argv[1]) != "large") ||
                      (std::string(argv[2]) != "1" && std::string(argv[2]) != "32") ||
                      (std::string(argv[3]) != "seq-first" && std::string(argv[3]) != "shuffle-first"))) {
            std::cerr << "Usage: gather_rows_order --plan\n"
                         "   or: gather_rows_order small|large 1|32 seq-first|shuffle-first\n";
            return 2;
        }
        int device = 0;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&device), "cudaGetDevice");
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        if (properties.l2CacheSize <= 0) {
            throw std::runtime_error("Device did not report a positive L2 capacity");
        }
        std::cerr << "gpu=" << properties.name << "\nl2_bytes=" << properties.l2CacheSize
                  << "\ndata_seed=" << data_seed << "\nshuffle_seed=" << order_seed << '\n';
        if (plan) {
            print_plan(properties.l2CacheSize);
        } else {
            run_pair(argv[1], std::string(argv[2]) == "1" ? 1u : 32u,
                     std::string(argv[3]) == "shuffle-first", properties.l2CacheSize);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

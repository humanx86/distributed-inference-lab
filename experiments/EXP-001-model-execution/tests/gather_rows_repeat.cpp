#include "gather_rows_cpu.hpp"
#include "gather_rows_cuda.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

int main() {
    try {
        constexpr std::size_t cols = 171;
        constexpr std::size_t warmup = 5;
        constexpr std::size_t measured = 20;

        Matrix input{2, cols, std::vector<float>(2 * cols)};
        for (std::size_t k = 0; k < input.data.size(); ++k) {
            input.data[k] = static_cast<float>(k + 1);
        }

        const std::vector<std::int64_t> indices{1, 0, 1};
        const Matrix expected = gather_rows_cpu(input, indices);

        std::vector<float> samples(warmup + measured);
        std::vector<GatherHostSample> host_samples(samples.size());
        const Matrix actual = gather_rows_cuda(input, indices, samples, host_samples);

        if (actual.rows != 3 || actual.cols != cols ||
            actual.data != expected.data) {
            std::cerr << "FAIL: repeated gather disagrees with CPU\n";
            return 1;
        }

        std::cout << "iteration,phase,device_event_us,"
                     "host_enqueue_us,host_synchronized_us\n";
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const char* phase = i < warmup ? "warmup" : "measured";
            std::cout << i << ',' << phase << ',' << samples[i] << ','
                      << host_samples[i].enqueue_us << ','
                      << host_samples[i].synchronized_us << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
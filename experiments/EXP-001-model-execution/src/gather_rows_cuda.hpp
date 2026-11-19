#pragma once

#include "gather_rows_cpu.hpp"

struct GatherHostSample {
    double enqueue_us;
    double synchronized_us;
};

Matrix gather_rows_cuda(const Matrix& input, 
                        const std::vector<std::int64_t>& indices,
                        std::vector<float>& device_event_us,
                        std::vector<GatherHostSample>& host_samples);


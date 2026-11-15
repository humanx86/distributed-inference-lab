#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct Matrix {
    std::size_t rows;
    std::size_t cols;
    std::vector<float> data;
};

Matrix gather_rows_cpu(
    const Matrix& input,
    const std::vector<std::int64_t>& indices
);

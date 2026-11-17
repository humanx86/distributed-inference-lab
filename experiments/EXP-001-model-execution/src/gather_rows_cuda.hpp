#pragma once

#include "gather_rows_cpu.hpp"

Matrix gather_rows_cuda(const Matrix& input, const std::vector<std::int64_t>& indices);
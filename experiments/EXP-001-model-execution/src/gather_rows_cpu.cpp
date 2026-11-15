#include "gather_rows_cpu.hpp"

#include <limits>
#include <stdexcept>


Matrix gather_rows_cpu( const Matrix& input, const std::vector<std::int64_t>& indices) {

    // Reject empty shapes or index lists
    // This also guarantees input.cols != 0 which the overflow checks below divide by
    if (input.rows == 0 || input.cols == 0 || indices.empty()) {
        throw std::invalid_argument("Empty matrix or index list");
    }
    
    // Check both size products
    // Ensure rows * cols and indices.size() * cols fit in size_t, so the
    // allocation size and every flat offset below are computed without wrapping
    const auto max_size = std::numeric_limits<std::size_t>::max();
    if (input.rows > max_size / input.cols) {
        throw std::overflow_error("Input shape product overflows");
    }
    if (indices.size() > max_size / input.cols) {
        throw std::overflow_error("Output shape product overflows");
    }

    // Check input storage
    // The buffer must hold exactly rows * cols elements; if not an offset
    // that is valid for the shape could still read past the end of data
    if (input.data.size() != input.rows * input.cols) {
        throw std::invalid_argument("Input data size does not match shape");
    }

    // Validate the original signed indices before any conversion: 
    // casting to size_t first would turn negatives into huge values, 
    // and truncate large values where size_t is 32-bit
    for (const std::int64_t index : indices) {
        if (index < 0 || static_cast<std::uint64_t>(index) >= input.rows) {
            throw std::out_of_range("Index out of range");
        }
    }
    
    // allocate output
    Matrix output{
        indices.size(),
        input.cols,
        std::vector<float>(indices.size() * input.cols)
    };

    // Copy elements; every offset is proven in bounds above
    // The largest possible offset is:
    // (input.rows - 1) * input.cols + (input.cols - 1) = input.rows * input.cols - 1
    for (std::size_t i = 0; i < output.rows; ++i) {
        const auto src_row = static_cast<std::size_t>(indices[i]);

        for (std::size_t j = 0; j < output.cols; ++j) {
            const std::size_t source_offset = src_row * input.cols + j;
            const std::size_t destination_offset = i * output.cols + j;

            output.data[destination_offset] = input.data[source_offset];
        }
    }

    return output;
}

#include "../src/gather_rows_cuda.hpp"

#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <limits>

namespace {

struct InvalidInputCase {
    const char* name;
    Matrix input;
    std::vector<std::int64_t> indices;
};

}  // namespace

int main() {
    try {
        // ---------------------------------------------------------------
        // Fixture 1: basic gather, hand-calculated
        //
        // Values encode position: tens digit = source row + 1, units digit =
        // column. A wrong value in the output shows directly which row and
        // column were read.
        //
        // Indices {2, 0, 2} cover:
        //   - out-of-order selection (2 before 0)
        //   - a repeated index (2 appears twice)
        //   - an unselected row (row 1), so a kernel that ignores the
        //     indices and copies the input unchanged will fail
        // ---------------------------------------------------------------
        const Matrix input{3, 3, {
            10, 11, 12,
            20, 21, 22,
            30, 31, 32
        }};
        const std::vector<std::int64_t> indices{2, 0, 2};

        const std::vector<float> expected{
            30, 31, 32,
            10, 11, 12,
            30, 31, 32
        };

        // Validate the CPU reference
        const Matrix cpu = gather_rows_cpu(input, indices);

        if (cpu.rows != 3 || cpu.cols != 3 || cpu.data != expected) {
            throw std::runtime_error(
                "CPU reference disagrees with hand-calculated fixture"
            );
        }

        const auto require_cuda = [](cudaError_t status) {
            if (status != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(status));
            }
        };

        // Log the device before launching any kernel, so a failure report
        // identifies the hardware. Flushed in case a later step crashes
        int device = 0;
        cudaDeviceProp properties{};

        require_cuda(cudaGetDevice(&device));
        require_cuda(cudaGetDeviceProperties(&properties, device));

        std::cout
            << "gpu=" << properties.name << '\n'
            << "compute_capability="
            << properties.major << '.' << properties.minor << '\n'
            << std::flush;

        const Matrix gpu = gather_rows_cuda(input, indices);

        if (gpu.rows != cpu.rows || gpu.cols != cpu.cols) {
            throw std::runtime_error("CUDA output shape mismatch");
        }
        
        // Exact float comparison; a gather only copies values
        if (gpu.data != cpu.data) {
            throw std::runtime_error("CUDA output values mismatch");
        }

        std::cout << "PASS: CUDA gather matches CPU and hand-calculated fixture\n";

        // ---------------------------------------------------------------
        // Fixture 2: thread-block boundary
        //
        // Output is 3 rows x 171 cols = 513 elements, one more than 512.
        // For any power-of-two block size up to 512, the last block has
        // exactly one in-range thread. This catches a launch that is one
        // block short, or a bounds check that drops the last element
        //
        // It does not reliably catch a missing bounds check: writes just
        // past the output buffer can land in allocation padding without
        // any error. We should run the test under compute-sanitizer for that
        //
        // Values start at 1 so no element is 0: an output element the kernel
        // never wrote cannot match by accident in a zero-initialized buffer
        // ---------------------------------------------------------------
        constexpr std::size_t boundary_cols = 171;

        Matrix boundary_input{
            2,
            boundary_cols,
            std::vector<float>(2 * boundary_cols)
        };

        for (std::size_t k = 0; k < boundary_input.data.size(); ++k) {
            boundary_input.data[k] = static_cast<float>(k + 1);
        }

        const std::vector<std::int64_t> boundary_indices{1, 0, 1};

        // CPU was validated in fixture 1, so it is the reference here
        const Matrix boundary_cpu = gather_rows_cpu(boundary_input, boundary_indices);
        const Matrix boundary_gpu = gather_rows_cuda(boundary_input, boundary_indices);

        if (boundary_cpu.rows != 3 ||
            boundary_cpu.cols != boundary_cols ||
            boundary_cpu.data.size() != 3 * boundary_cols) {
            throw std::runtime_error("CPU boundary output shape/storage mismatch");
        }

        if (boundary_gpu.rows != 3 ||
            boundary_gpu.cols != boundary_cols ||
            boundary_gpu.data.size() != 3 * boundary_cols) {
            throw std::runtime_error("CUDA boundary output shape/storage mismatch");
        }

        if (boundary_gpu.data != boundary_cpu.data) {
            throw std::runtime_error("CUDA boundary output values mismatch");
        }

        std::cout << "PASS: CUDA gather 513-element boundary fixture\n";

        // ---------------------------------------------------------------
        // Fixture 3: single input row, repeated index
        //
        // Smallest possible input (1 row), and the output has more rows than
        // the input. This catches code that sizes the output from
        // input.rows instead of indices.size(), or that assumes indices
        // are unique
        // ---------------------------------------------------------------
        const Matrix single_input{1, 3, {7, 8, 9}};
        const std::vector<std::int64_t> single_indices{0, 0};

        const std::vector<float> single_expected{
            7, 8, 9,
            7, 8, 9
        };

        const Matrix single_cpu = gather_rows_cpu(single_input, single_indices);
        const Matrix single_gpu = gather_rows_cuda(single_input, single_indices);

        // Both implementations are checked against the hand-calculated
        // values, not just against each other
        if (single_cpu.rows != 2 || single_cpu.cols != 3) {
            throw std::runtime_error("CPU single-row output shape mismatch");
        }

        if (single_gpu.rows != 2 || single_gpu.cols != 3) {
            throw std::runtime_error("CUDA single-row output shape mismatch");
        }

        if (single_cpu.data != single_expected) {
            throw std::runtime_error("CPU single-row output values mismatch");
        }

        if (single_gpu.data != single_expected) {
            throw std::runtime_error("CUDA single-row output values mismatch");
        }

        std::cout << "PASS: CUDA gather single-row repeated-index fixture\n";

        // ---------------------------------------------------------------
        // Rejection tests: invalid indices must be caught on the host
        // ---------------------------------------------------------------
        // Negative index: catches a signed/unsigned mix-up where -1 wraps to
        // a huge value, or validation that only checks the upper bound
        const std::vector<std::int64_t> negative_index{-1};

        try {
            gather_rows_cuda(input, negative_index);
            std::cerr << "FAIL: CUDA gather accepted negative index -1\n";
            return 1;
        } catch (const std::out_of_range&) {
            std::cout << "PASS: CUDA gather rejects negative index\n";
        }

        // Index == rows (3 for the 3x3 input): the first out-of-range value
        // Catches an off-by-one check (index > rows instead of index >= rows)
        const std::vector<std::int64_t> past_end_index{3};

        try {
            gather_rows_cuda(input, past_end_index);
            std::cerr << "FAIL: CUDA gather accepted out-of-range index 3\n";
            return 1;
        } catch (const std::out_of_range&) {
            std::cout << "PASS: CUDA gather rejects index equal to row count\n";
        }

        // ---------------------------------------------------------------
        // Rejection tests: empty inputs and inconsistent storage
        //
        // Each case should have exactly one defect, so a pass shows that
        // the specific rule is enforced. Only std::invalid_argument counts
        // as a pass. Any other exception reaches the outer handler, and
        // returning normally is also a failure
        // ---------------------------------------------------------------
        const std::vector<InvalidInputCase> invalid_input_cases{
            // Any index is out of range when rows == 0, so this case also
            // requires the shape to be validated before the indices.
            // Otherwise the wrapper throws std::out_of_range and this fails
            {"zero rows", Matrix{0, 3, {}}, {0}},

            // Storage is consistent (0 == 3 * 0) and index 0 is in range,
            // so the zero width is the only defect
            {"zero columns", Matrix{3, 0, {}}, {0}},

            // Valid 3x3 input; the empty index list is the only defect
            {"empty indices", input, {}},

            // Declared 3x3 needs 9 values, but only 8 are stored. Index 2 is
            // valid for the declared shape, but row 2 spans elements 6..8 and
            // element 8 does not exist. Without a storage-size check, the
            // wrapper would read past the end of the host buffer
            {"storage shorter than declared shape",
             Matrix{3, 3, {10, 11, 12, 20, 21, 22, 30, 31}}, {2}},

            // 10 values for a 3x3 shape. A check that only rejects short
            // storage (size < rows * cols) would let this through
            {"storage longer than declared shape",
             Matrix{3, 3, {10, 11, 12, 20, 21, 22, 30, 31, 32, 99}}, {0}},
        };

        for (const InvalidInputCase& test_case : invalid_input_cases) {
            try {
                (void)gather_rows_cuda(test_case.input, test_case.indices);
                std::cerr << "FAIL: CUDA gather accepted invalid input: "
                          << test_case.name << '\n';
                return 1;
            } catch (const std::invalid_argument&) {
                std::cout << "PASS: CUDA gather rejects " << test_case.name << '\n';
            }
        }

        // ---------------------------------------------------------------
        // Rejection tests: shape-size overflow
        //
        // The huge dimensions are metadata only; each vector holds one or
        // two floats. over_half * 2 == 2^N for an N-bit size_t, which wraps
        // to exactly 0, so an unchecked multiply sees an empty matrix
        //
        // The storage deliberately disagrees with the declared shape, so
        // these cases also require the overflow checks to run before
        // storage validation. If the storage check runs first, it throws
        // std::invalid_argument, which reaches the outer handler and fails
        //
        // Not covered: large GPU allocations, or overflow when converting
        // element counts to byte counts (count * sizeof(float))
        // ---------------------------------------------------------------
        const auto max_size = std::numeric_limits<std::size_t>::max();
        const auto over_half = max_size / 2 + 1;

        const std::vector<InvalidInputCase> overflow_inputs{
            // rows * cols = over_half * 2 cannot fit in std::size_t
            {"input shape overflow", {over_half, 2, {7, 8}}, {0}},

            // The input shape (1 * over_half) fits, so only the output
            // shape, indices.size() * cols = 2 * over_half, overflows.
            // The storage mismatch (1 value vs over_half) is real here, so
            // the wrapper must check the output size before the storage
            {"output shape overflow", {1, over_half, {7}}, {0, 0}},
        };

        for (const InvalidInputCase& test_case : overflow_inputs) {
            try {
                (void)gather_rows_cuda(test_case.input, test_case.indices);
                std::cerr << "FAIL: CUDA gather accepted overflowing shape: "
                          << test_case.name << '\n';
                return 1;
            } catch (const std::overflow_error&) {
                std::cout << "PASS: CUDA gather rejects " << test_case.name << '\n';
            }
        }

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
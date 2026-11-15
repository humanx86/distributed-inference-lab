#include "../src/gather_rows_cpu.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <limits>

int main() {
    try {
        const Matrix input{3, 3, {
            10, 11, 12,
            20, 21, 22,
            30, 31, 32
        }};
        const std::vector<std::int64_t> indices{2, 0, 2};

        const Matrix output = gather_rows_cpu(input, indices);

        // Hand calculated expected output
        const std::vector<float> expected{
            30, 31, 32,
            10, 11, 12,
            30, 31, 32
        };

        if (output.rows != 3 || output.cols != 3) {
            std::cerr << "FAIL: unexpected output shape\n";
            return 1;
        }

        if (output.data != expected) {
            std::cerr << "FAIL: unexpected output values\n";
            return 1;
        }

        std::cout << "PASS: gather_rows_cpu repeated-index fixture\n";
        
        const std::vector<std::int64_t> indices_more= {1, 0, 1, 2};
        const Matrix output_more = gather_rows_cpu(input, indices_more);

        const std::vector<float> expected_more{
            20, 21, 22,
            10, 11, 12,
            20, 21, 22,
            30, 31, 32
        };

                if (output_more.rows != 4 || output_more.cols != 3) {
            std::cerr << "FAIL: unexpected output-row-count shape\n";
            return 1;
        }

        if (output_more.data != expected_more) {
            std::cerr << "FAIL: unexpected output-row-count values\n";
            return 1;
        }

        std::cout << "PASS: gather_rows_cpu output-row-count fixture\n";

        try {
            (void)gather_rows_cpu(input, {-1});

            std::cerr << "FAIL: negative index was accepted\n";
            return 1;
        } catch (const std::out_of_range&) {
            std::cout << "PASS: negative index rejected\n";
        }

        try {
            (void)gather_rows_cpu(input, {3});

            std::cerr << "FAIL: upper-bound index was accepted\n";
            return 1;
        } catch (const std::out_of_range&) {
            std::cout << "PASS: upper-bound index rejected\n";
        }

        struct InvalidInputCase {
            const char* name;
            Matrix input;
            std::vector<std::int64_t> indices;
        };

        const std::vector<InvalidInputCase> invalid_inputs{
            {"zero rows", {0, 3, {}}, {0}},
            {"zero cols", {3, 0, {}}, {0}},
            {"empty indices", {3, 3, {10, 11, 12, 20, 21, 22, 30, 31, 32}}, {}},
            // Index 2 is valid for the shape, but row 2 would read past the end of 8 elements
            {"data shorter than shape", {3, 3, {10, 11, 12, 20, 21, 22, 30, 31}}, {2}},
            {"data longer than shape", {3, 3, {10, 11, 12, 20, 21, 22, 30, 31, 32, 99}}, {0}},
        };

        for (const auto& test_case : invalid_inputs) {
            try {
                (void)gather_rows_cpu(test_case.input, test_case.indices);

                std::cerr << "FAIL: accepted " << test_case.name << '\n';
                return 1;
            } catch (const std::invalid_argument&) {
                std::cout << "PASS: rejected " << test_case.name << '\n';
            } catch (const std::exception& error) {
                std::cerr << "FAIL: " << test_case.name
                          << " threw the wrong exception type: "
                          << error.what() << '\n';
                return 1;
            }
        }

        const Matrix single_input{1, 3, {7, 8, 9}};
        const auto single_before = single_input.data;

        Matrix single_output = gather_rows_cpu(single_input, {0, 0});

        // Hand calculated expected output: row 0 twice
        const std::vector<float> expected_single{
            7, 8, 9,
            7, 8, 9
        };

        if (single_output.rows != 2 || single_output.cols != 3) {
            std::cerr << "FAIL: unexpected single-row output shape\n";
            return 1;
        }

        if (single_output.data != expected_single) {
            std::cerr << "FAIL: unexpected single-row output values\n";
            return 1;
        }

        std::cout << "PASS: single-row input fixture\n";

        // Writing to the output must not change the input
        single_output.data[0] = -999.0f;

        if (single_input.data != single_before) {
            std::cerr << "FAIL: output storage aliases input storage\n";
            return 1;
        }

        std::cout << "PASS: output storage is independent\n";

        const auto max_size = std::numeric_limits<std::size_t>::max();
        const auto over_half = max_size / 2 + 1;

        const std::vector<InvalidInputCase> overflow_inputs{
            {"input shape overflow", {over_half, 2, {7, 8}}, {0}},
            // 1 * over_half fits, but 2 selected rows * over_half overflows
            {"output shape overflow", {1, over_half, {7}}, {0, 0}},
        };

        for (const auto& test_case : overflow_inputs) {
            try {
                (void)gather_rows_cpu(test_case.input, test_case.indices);

                std::cerr << "FAIL: accepted " << test_case.name << '\n';
                return 1;
            } catch (const std::overflow_error&) {
                std::cout << "PASS: rejected " << test_case.name << '\n';
            } catch (const std::exception& error) {
                std::cerr << "FAIL: " << test_case.name
                          << " threw the wrong exception type: "
                          << error.what() << '\n';
                return 1;
            }
        }

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: "
                  << error.what() << '\n';
        return 1;
    }
}

#include "gather_rows_cuda.hpp"

#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&ptr_, bytes), "cudaMalloc");
    }

    // Fallback cleanup
    ~DeviceBuffer() noexcept {
        if (ptr_ != nullptr) {
            const auto status = cudaFree(ptr_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "cudaFree during cleanup: %s\n",
                    cudaGetErrorString(status));
            }
        }
    }

    // Prevent two objects from owning the same allocation
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const noexcept {
        return ptr_;
    }

    // Use on the normal path so cleanup errors propagate
    void release_checked() {
        if (ptr_ != nullptr) {
            void* allocation = ptr_;
            ptr_ = nullptr;
            check_cuda(cudaFree(allocation), "cudaFree");
        }
    }

private:
    void* ptr_ = nullptr;
};

std::size_t validate_gather_inputs( const Matrix& input,
                                    const std::vector<std::int64_t>& indices ) {

    if (input.rows == 0 || input.cols == 0 || indices.empty()) {
        throw std::invalid_argument("Empty matrix or index list");
    }

    const auto max_size = std::numeric_limits<std::size_t>::max();

    if (input.rows > max_size / input.cols) {
        throw std::overflow_error("Input shape product overflows");
    }

    if (indices.size() > max_size / input.cols) {
        throw std::overflow_error("Output shape product overflows");
    }

    const std::size_t input_elements = input.rows * input.cols;
    const std::size_t output_elements = indices.size() * input.cols;

    if (input.data.size() != input_elements) {
        throw std::invalid_argument("Input data size does not match shape");
    }

    for (const std::int64_t index : indices) {
        if (index < 0 ||
            static_cast<std::uint64_t>(index) >= input.rows) {
            throw std::out_of_range("Index out of range");
        }
    }

    if (input_elements > max_size / sizeof(float) ||
        indices.size() > max_size / sizeof(std::int64_t) ||
        output_elements > max_size / sizeof(float)) {
        throw std::overflow_error("Buffer byte count overflows");
    }

    return output_elements;
}

}  // namespace


__global__ void gather_all_rows_kernel( const float* input,
                                        const std::int64_t* indices,    // signed values, following PyTorch index tensors
                                        float* output,
                                        std::size_t out_rows,
                                        std::size_t cols) {

    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = out_rows * cols;

    if (t >= total) {
        return;
    }

    const std::size_t i = t / cols;
    const std::size_t j = t % cols;
    const std::size_t src_row = static_cast<std::size_t>(indices[i]);

    output[t] = input[src_row * cols + j];
}


Matrix gather_rows_cuda( const Matrix& input, 
                         const std::vector<std::int64_t>& indices ) {

    const std::size_t total = validate_gather_inputs(input, indices);

    const std::size_t out_rows = indices.size();
    const std::size_t cols = input.cols;

    const std::size_t input_bytes = input.data.size() * sizeof(float);

    const std::size_t indices_bytes = indices.size() * sizeof(std::int64_t);

    const std::size_t output_bytes = total * sizeof(float);

    constexpr unsigned int threads_per_block = 256;

    const std::size_t blocks_needed = total / threads_per_block + (total % threads_per_block != 0);

    int device = 0;
    cudaDeviceProp properties{};

    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    if (threads_per_block > static_cast<unsigned int>(properties.maxThreadsPerBlock) ||
        threads_per_block > static_cast<unsigned int>(properties.maxThreadsDim[0])) {
        throw std::runtime_error("Unsupported thread block size");
    }

    if (blocks_needed > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::length_error("Output exceeds supported 1D grid size");
    }

    const unsigned int blocks = static_cast<unsigned int>(blocks_needed);

    // CPU storage that will receive the GPU result
    Matrix output{
        out_rows,
        cols,
        std::vector<float>(total)
    };

    DeviceBuffer d_input(input_bytes);
    DeviceBuffer d_indices(indices_bytes);
    DeviceBuffer d_output(output_bytes);

    check_cuda(cudaMemcpy(d_input.get(), input.data.data(), 
               input_bytes, cudaMemcpyHostToDevice), "copy input to GPU");

    check_cuda(cudaMemcpy(d_indices.get(), indices.data(), 
               indices_bytes, cudaMemcpyHostToDevice), "copy indices to GPU");

    check_cuda(cudaGetLastError(), "CUDA state before gather launch");

    gather_all_rows_kernel<<<blocks, threads_per_block>>>(
        static_cast<const float*>(d_input.get()),
        static_cast<const std::int64_t*>(d_indices.get()),
        static_cast<float*>(d_output.get()),
        out_rows,
        cols);

    check_cuda(cudaGetLastError(), "gather kernel launch");
    check_cuda(cudaDeviceSynchronize(), "gather kernel execution");

    check_cuda(cudaMemcpy(output.data.data(), d_output.get(),
               output_bytes, cudaMemcpyDeviceToHost), "copy output to CPU");

    d_output.release_checked();
    d_indices.release_checked();
    d_input.release_checked();

    return output;

}
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>


#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err_ = (call);                                    \
        if (err_ != cudaSuccess) {                                    \
            std::cerr << "CUDA error at "                             \
                      << __FILE__ << ":" << __LINE__ << ": "          \
                      << cudaGetErrorString(err_) << '\n';            \
            return 1;                                                 \
        }                                                             \
    } while (0)


// RAII owner for one device allocation
// Construction acquires GPU memory, destruction releases it
// Copy is disabled to prevent two objects from owning the same allocation
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) 
        : ptr_(nullptr)
    {
        cudaError_t err = cudaMalloc(&ptr_, bytes);

        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    ~DeviceBuffer() noexcept
    {
        if (ptr_ != nullptr) {
            cudaError_t err = cudaFree(ptr_);

            if (err != cudaSuccess) {
                std::cerr << "cudaFree failed: "
                          << cudaGetErrorString(err) << "\n";
            }
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get()
    {
        return ptr_;
    }

private:
    float* ptr_;
};

__global__ void vector_add (const float *A, 
                            const float *B, 
                            float *C, 
                            int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }

}

int run_case(int N) {
    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N);

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = 2.f * static_cast<float>(i);
    }

    const size_t bytes = N * sizeof(float);

    DeviceBuffer d_A(bytes);
    DeviceBuffer d_B(bytes);
    DeviceBuffer d_C(bytes);
    
    CUDA_CHECK(cudaMemcpy(d_A.get(), A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B.get(), B.data(), bytes, cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    vector_add<<<blocks, threads_per_block>>>(d_A.get(), d_B.get(), d_C.get(), N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C.data(), d_C.get(), bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        const float expected = A[i] + B[i];

        if (!std::isfinite(C[i]) || C[i] != expected) {
            std::cerr << "Mismatch at N = " << N
                      << ", i = " << i
                      << ": expected " << expected
                      << ", got " << C[i] << "\n";

            return 1;
        }
    }

    return 0;
}

int main() {
    try {
        const std::vector<int> sizes = { 1, 255, 256, 257, 1000, 1024};

        for (int N : sizes) {
            std::cout << "Testing N = " << N << "...";
            if (run_case(N) != 0) {
                std::cout << "FAIL\n";
                return 1;
            }
            std::cout << "PASS\n";
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// Original teaching harness
// Build: nvcc -O3 -lineinfo -std=c++17 gpu_lab.cu -o gpu_lab
// Optional NVTX: add -DLAB_USE_NVTX -I/path/to/NVTX/c/include -ldl
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef LAB_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

void check(cudaError_t e, const char* expression) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(e));
}
#define CUDA_OK(x) check((x), #x)
struct Range {
    explicit Range(const char* name) {
#ifdef LAB_USE_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~Range() {
#ifdef LAB_USE_NVTX
        nvtxRangePop();
#endif
    }
};
struct Buffer {
    float* p = nullptr;
    explicit Buffer(size_t n) {
        CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), n)*sizeof(float)));
    }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t s{};
    Stream() { CUDA_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(s); }
};
struct Events {
    cudaEvent_t a{}, b{};
    Events() { CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&b)); }
    ~Events() { cudaEventDestroy(a); cudaEventDestroy(b); }
};

// Uniform grid-stride loop: adjacent lanes access adjacent floats each iteration
__global__ void elementwise(const float* x, const float* y, float* z, size_t n, int op) {
    for (size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
         i < n; i += size_t(gridDim.x)*blockDim.x) {
        z[i] = op == 0 ? x[i] : op == 1 ? x[i] + y[i] : fmaf(2.0f, x[i], 1.0f);
    }
}

// Intentionally traffic/launch-heavy but correct global-memory tree
__global__ void pair_stage(const float* x, float* partial, size_t n) {
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    size_t j = 2*i;
    if (j < n) partial[i] = x[j] + (j+1 < n ? x[j+1] : 0.0f);
}

// Precondition: blockDim.x is a power of two. Every thread reaches every barrier
__global__ void tree_stage(const float* x, float* partial, size_t n) {
    extern __shared__ float shared[];
    unsigned t = threadIdx.x;
    float sum = 0.0f;
    for (size_t i = size_t(blockIdx.x)*blockDim.x+t;
         i < n; i += size_t(gridDim.x)*blockDim.x) sum += x[i];
    shared[t] = sum; // zero is the identity for threads with no input
    __syncthreads();
    for (unsigned stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (t < stride) shared[t] += shared[t+stride];
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = shared[0];
}

// All 32 lanes call this function. No partial-mask or divergent caller is allowed
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v; // the complete sum is meaningful in lane zero
}

// Precondition: block size is a multiple of 32, at most 1024; no early returns
__global__ void warp_stage(const float* x, float* partial, size_t n) {
    __shared__ float warp_totals[32];
    unsigned t = threadIdx.x, lane = t % 32, warp = t / 32;
    float sum = 0.0f;
    for (size_t i = size_t(blockIdx.x)*blockDim.x+t;
         i < n; i += size_t(gridDim.x)*blockDim.x) sum += x[i];
    sum = warp_sum(sum);
    if (lane == 0) warp_totals[warp] = sum;
    __syncthreads(); // publishes every warp's partial before warp zero reads
    if (warp == 0) {
        float v = lane < blockDim.x/32 ? warp_totals[lane] : 0.0f;
        v = warp_sum(v); // all 32 lanes of warp zero participate
        if (lane == 0) partial[blockIdx.x] = v;
    }
}

struct Options {
    std::string op = "add", variant = "tree", dataset = "dyadic";
    size_t n = 1 << 20;
    int block = 256, warmup = 5, samples = 10, batch = 10, gap_us = 0;
    double atol = 1e-4, rtol = 1e-4;
    bool check_only = false;
};
int parse_count(const std::string& value, const std::string& option) {
    if (value.empty() || !std::all_of(value.begin(), value.end(),
                                    [](char c) { return c >= '0' && c <= '9'; }))
        throw std::runtime_error("Use decimal digits for " + option);
    size_t used = 0;
    unsigned long long count = std::stoull(value, &used, 10);
    if (used != value.size() || count > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Out-of-range count for " + option);
    return static_cast<int>(count);
}
double parse_tolerance(const std::string& value, const std::string& option) {
    size_t used = 0;
    double tolerance = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(tolerance) || tolerance < 0)
        throw std::runtime_error("Invalid tolerance for " + option);
    return tolerance;
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--check-only") { o.check_only = true; continue; }
        if (k == "--help") {
            std::cout << "--op copy|add|transform|reduce --variant pair|tree|warp|cub\n"
                      << "--n N --block 128|256|512 --warmup W --samples S --batch B\n"
                      << "--dataset dyadic|random|ones --gap-us U --atol A --rtol R\n"
                      << "--check-only (all operations/variants, small boundary inputs)\n"
                      << "Integer sizes/counts use decimal digits: 1000000, not 1e6.\n";
            std::exit(0);
        }
        if (++i == argc) throw std::runtime_error("Missing value for " + k);
        std::string v = argv[i];
        if (k == "--op") o.op = v;
        else if (k == "--variant") o.variant = v;
        else if (k == "--dataset") o.dataset = v;
        else if (k == "--n") o.n = parse_count(v, k);
        else if (k == "--block") o.block = parse_count(v, k);
        else if (k == "--warmup") o.warmup = parse_count(v, k);
        else if (k == "--samples") o.samples = parse_count(v, k);
        else if (k == "--batch") o.batch = parse_count(v, k);
        else if (k == "--gap-us") o.gap_us = parse_count(v, k);
        else if (k == "--atol") o.atol = parse_tolerance(v, k);
        else if (k == "--rtol") o.rtol = parse_tolerance(v, k);
        else throw std::runtime_error("Unknown option " + k);
    }
    if (o.op!="copy" && o.op!="add" && o.op!="transform" && o.op!="reduce") throw std::runtime_error("Invalid op");
    if (o.variant!="pair" && o.variant!="tree" && o.variant!="warp" && o.variant!="cub") throw std::runtime_error("Invalid variant");
    if (o.dataset!="dyadic" && o.dataset!="random" && o.dataset!="ones") throw std::runtime_error("Invalid dataset");
    if (o.block!=128 && o.block!=256 && o.block!=512) throw std::runtime_error("Use block 128, 256 or 512");
    if (o.n > size_t(std::numeric_limits<int>::max()) || o.warmup<0 || o.samples<1 || o.batch<1 || o.gap_us<0)
        throw std::runtime_error("Invalid size/count");
    if (!std::isfinite(o.atol) || !std::isfinite(o.rtol) || o.atol<0 || o.rtol<0) throw std::runtime_error("Invalid tolerance");
    if (!o.check_only && o.n==0) throw std::runtime_error("N=0 belongs in correctness tests, not this timing study");
    return o;
}

struct Work {
    Options o;
    cudaDeviceProp prop{};
    Stream stream;
    Buffer x, y, z, p, q, result;
    void* cub_temp = nullptr;
    size_t cub_bytes = 0;
    std::vector<float> hx, hy, got;
    std::vector<double> ref;
    explicit Work(Options a): o(a), x(a.n), y(a.n), z(a.n), p(a.n), q(a.n), result(1),
        hx(a.n), hy(a.n), got(a.op=="reduce" ? 1 : a.n), ref(got.size()) {
        CUDA_OK(cudaGetDeviceProperties(&prop, 0));
        if (o.block > prop.maxThreadsPerBlock || prop.warpSize != 32) throw std::runtime_error("Unsupported block/warp size");
        uint32_t state = 123456789u;
        for (size_t i=0; i<o.n; ++i) {
            state ^= state<<13; state ^= state>>17; state ^= state<<5;
            hx[i] = o.dataset=="ones" ? 1.0f : o.dataset=="dyadic" ? float(int(i%17)-8)/16.0f
                    : float(int(state & 65535u)-32768)/32768.0f;
            hy[i] = float(int(i%13)-6)/16.0f;
        }
        if (o.op=="reduce") { double s=0; for (float v:hx) s+=double(v); ref[0]=s; }
        else for (size_t i=0; i<o.n; ++i)
            ref[i]=o.op=="copy" ? double(hx[i]) : o.op=="add" ? double(hx[i])+double(hy[i]) : 2.0*double(hx[i])+1.0;
        if (o.n) {
            Range range("H2D setup");
            CUDA_OK(cudaMemcpy(x.p, hx.data(), o.n*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(y.p, hy.data(), o.n*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_OK(cudaDeviceSynchronize()); // complete setup before any consumer; outside timing
        }
        if (o.op=="reduce" && o.variant=="cub" && o.n) {
            CUDA_OK(cub::DeviceReduce::Sum(nullptr, cub_bytes, x.p, result.p, int(o.n), stream.s));
            CUDA_OK(cudaMalloc(&cub_temp, std::max(size_t(1),cub_bytes)));
        }
    }
    ~Work() { if (cub_temp) cudaFree(cub_temp); }
    unsigned blocks(size_t n) const {
        return unsigned(std::min((n+o.block-1)/o.block, size_t(prop.multiProcessorCount)*4));
    }
    void enqueue() {
        if (o.op!="reduce") {
            if (o.n) {
                int op=o.op=="copy" ? 0 : o.op=="add" ? 1 : 2;
                elementwise<<<blocks(o.n),o.block,0,stream.s>>>(x.p,y.p,z.p,o.n,op);
                CUDA_OK(cudaGetLastError());
            }
            return;
        }
        if (o.n==0) { CUDA_OK(cudaMemsetAsync(result.p,0,sizeof(float),stream.s)); return; }
        if (o.variant=="cub") {
            CUDA_OK(cub::DeviceReduce::Sum(cub_temp,cub_bytes,x.p,result.p,int(o.n),stream.s));
            CUDA_OK(cudaGetLastError()); return;
        }
        size_t m=o.n;
        const float* src=x.p;
        bool use_p=true;
        do {
            size_t count=o.variant=="pair" ? (m+1)/2 : blocks(m);
            float* dst=count==1 ? result.p : use_p ? p.p : q.p;
            if (o.variant=="pair")
                pair_stage<<<unsigned((count+o.block-1)/o.block),o.block,0,stream.s>>>(src,dst,m);
            else if (o.variant=="tree")
                tree_stage<<<unsigned(count),o.block,o.block*sizeof(float),stream.s>>>(src,dst,m);
            else warp_stage<<<unsigned(count),o.block,0,stream.s>>>(src,dst,m);
            CUDA_OK(cudaGetLastError());
            src=dst; m=count; use_p=!use_p;
        } while (m>1);
    }
    double validate() {
        Range range("D2H correctness");
        CUDA_OK(cudaStreamSynchronize(stream.s));
        if (!got.empty()) CUDA_OK(cudaMemcpy(got.data(),o.op=="reduce" ? result.p:z.p,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
        double worst=0;
        for (size_t i=0;i<got.size();++i) {
            double e=std::abs(double(got[i])-ref[i]);
            worst=std::max(worst,e);
            if (!std::isfinite(got[i]) || e > o.atol + o.rtol*std::abs(ref[i]))
                throw std::runtime_error("Correctness failed at index " + std::to_string(i)+
                    ", got="+std::to_string(got[i])+", ref="+std::to_string(ref[i]));
        }
        return worst;
    }
};

int main(int argc,char** argv) {
    try {
        Options o=parse(argc,argv);
        CUDA_OK(cudaSetDevice(0));
        cudaDeviceProp prop{}; int driver=0,runtime=0;
        CUDA_OK(cudaGetDeviceProperties(&prop,0));
        CUDA_OK(cudaDriverGetVersion(&driver)); CUDA_OK(cudaRuntimeGetVersion(&runtime));
        std::cerr << "gpu="<<prop.name<<" cc="<<prop.major<<'.'<<prop.minor<<" SMs="<<prop.multiProcessorCount
                  <<" L2_bytes="<<prop.l2CacheSize<<" memory_bytes="<<prop.totalGlobalMem
                  <<" driver_api="<<driver<<" runtime_api="<<runtime<<" CUB_VERSION="<<CUB_VERSION<<'\n';
#ifdef LAB_USE_NVTX
        std::cerr << "nvtx=enabled\n";
#else
        std::cerr << "nvtx=disabled (rebuild with LAB_USE_NVTX for the annotated-trace exercise)\n";
#endif
        if (o.check_only) {
            for (size_t n: {size_t(0),size_t(1),size_t(7),size_t(31),size_t(32),size_t(33),size_t(127),size_t(128),size_t(129),size_t(255),size_t(256),size_t(257),size_t(511),size_t(512),size_t(513),size_t(1003),size_t(4099)})
                for (std::string dataset:{"dyadic","random","ones"})
                    for (std::string op:{"copy","add","transform","reduce"})
                        for (std::string variant:{"pair","tree","warp","cub"}) {
                            if (op!="reduce" && variant!="tree") continue;
                            Options c=o; c.n=n; c.dataset=dataset; c.op=op; c.variant=variant;
                            Work w(c); w.enqueue(); w.validate();
                        }
            std::cout << "PASS boundary suite at block="<<o.block<<"; this is not a proof for arbitrary inputs\n";
            return 0;
        }
        Work w(o); Events ev;
        { Range range("warmup"); for (int k=0;k<o.warmup;++k) w.enqueue(); }
        w.enqueue(); w.validate(); // one untimed correctness call, even if warmup=0
        std::cout << "sample,n,op,variant,block,warmup,batch,gap_us,dataset,event_span_ms,host_submit_ms,host_complete_ms,event_ms_per_op,max_abs_error\n";
        std::cout << std::setprecision(10);
        for (int sample=0;sample<o.samples;++sample) {
            CUDA_OK(cudaStreamSynchronize(w.stream.s));
            using Clock=std::chrono::steady_clock;
            auto t0=Clock::now();
            double submit=0, complete=0;
            float elapsed=0;
            {
                Range range("measured batch");
                CUDA_OK(cudaEventRecord(ev.a,w.stream.s));
                auto submit0=Clock::now();
                for (int k=0;k<o.batch;++k) {
                    if (o.gap_us) std::this_thread::sleep_for(std::chrono::microseconds(o.gap_us));
                    w.enqueue();
                }
                auto submit1=Clock::now();
                CUDA_OK(cudaEventRecord(ev.b,w.stream.s));
                CUDA_OK(cudaEventSynchronize(ev.b));
                auto t1=Clock::now();
                CUDA_OK(cudaEventElapsedTime(&elapsed,ev.a,ev.b));
                submit=std::chrono::duration<double,std::milli>(submit1-submit0).count();
                complete=std::chrono::duration<double,std::milli>(t1-t0).count();
            }
            double error=w.validate(); // OUTSIDE the measured NVTX range and all stored spans
            std::cout << sample<<','<<o.n<<','<<o.op<<','<<(o.op=="reduce"?o.variant:"na")<<','<<o.block<<','<<o.warmup
                <<','<<o.batch<<','<<o.gap_us<<','<<o.dataset<<','<<elapsed<<','<<submit<<','<<complete<<','<<elapsed/o.batch<<','<<error<<'\n'<<std::flush;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; return 2; }
}

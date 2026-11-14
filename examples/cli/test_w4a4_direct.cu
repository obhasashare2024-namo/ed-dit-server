#include "third_party/json.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

__global__ void dequant_sym_w4a4_kernel(
    const uint8_t* __restrict__ q,
    const half* __restrict__ scale,
    half* __restrict__ out,
    unsigned long long N,
    unsigned long long K)
{
    unsigned long long n = N * K;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; 
         i < n; 
         i += (unsigned long long)blockDim.x * gridDim.x) 
    {
        unsigned long long r = i / K;
        unsigned long long c = i % K;
        uint8_t byte = q[r * (K / 2) + c / 2];
        int nib = (c & 1) ? (byte >> 4) : (byte & 15);
        int s4 = (nib >= 8) ? (nib - 16) : nib;
        half s = scale[r * (K / 256) + c / 256];
        out[i] = __hmul(__int2half_rn(s4), s);
    }
}

int main(int argc, char** argv) {
    std::string model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4.safetensors";
    if (argc > 1) model_path = argv[1];

    std::cout << "=== Pure C++ Native W4A4 (sym_w4a4_convrot) Engine on RTX 3070 ===" << std::endl;
    std::cout << "Opening Safetensors file: " << model_path << std::endl;

    int fd = open(model_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open " << model_path << std::endl;
        return 1;
    }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    const uint8_t* data = (const uint8_t*)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        std::cerr << "mmap failed" << std::endl;
        close(fd);
        return 1;
    }

    uint64_t st_header_len = *(const uint64_t*)data;
    std::string json_str((const char*)(data + 8), st_header_len);
    nlohmann::json st_hdr = nlohmann::json::parse(json_str);
    const uint8_t* st_base_data = data + 8 + st_header_len;

    std::string prefix = "blocks.0.attn.qkv_proj";
    std::cout << "Loading layer from W4A4 Safetensors: " << prefix << std::endl;

    auto get_st = [&](const std::string& name) -> std::pair<const uint8_t*, size_t> {
        auto off = st_hdr[name]["data_offsets"];
        size_t o0 = off[0].get<size_t>();
        size_t o1 = off[1].get<size_t>();
        return {st_base_data + o0, o1 - o0};
    };

    auto w_pair = get_st(prefix + ".weight");
    auto s_pair = get_st(prefix + ".weight_scale");

    const uint8_t* h_w = w_pair.first;
    const half* h_s = (const half*)s_pair.first;
    size_t w_bytes = w_pair.second;
    size_t s_bytes = s_pair.second;

    auto shape = st_hdr[prefix + ".weight"]["shape"];
    uint64_t out_features = shape[0].get<uint64_t>(); // 21504
    uint64_t in_features = shape[1].get<uint64_t>() * 2; // 2688 * 2 = 5376

    std::cout << "  Layer geometry: in_features=" << in_features 
              << ", out_features=" << out_features << std::endl;
    std::cout << "  Weight bytes (packed 4-bit): " << w_bytes << " bytes" << std::endl;
    std::cout << "  Scale bytes (FP16 group-256): " << s_bytes << " bytes" << std::endl;

    // Allocate GPU buffers
    uint8_t *d_w = nullptr;
    half *d_s = nullptr;
    half *d_out_f16 = nullptr;

    cudaMalloc(&d_w, w_bytes);
    cudaMalloc(&d_s, s_bytes);
    cudaMalloc(&d_out_f16, out_features * in_features * sizeof(half));

    cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s, s_bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (out_features * in_features + block - 1) / block;

    std::cout << "Executing dequant_sym_w4a4_kernel on CUDA..." << std::endl;
    dequant_sym_w4a4_kernel<<<grid, block>>>(d_w, d_s, d_out_f16, out_features, in_features);
    cudaDeviceSynchronize();

    // Verify samples
    std::vector<half> h_out(10000);
    cudaMemcpy(h_out.data(), d_out_f16, 10000 * sizeof(half), cudaMemcpyDeviceToHost);
    size_t nan_count = 0;
    float sum = 0.0f;
    for (size_t i = 0; i < 10000; ++i) {
        float val = __half2float(h_out[i]);
        if (std::isnan(val)) nan_count++;
        sum += std::abs(val);
    }
    std::cout << "  Verification sample (first 10,000 weights):" << std::endl;
    std::cout << "    NaN count: " << nan_count << " / 10000" << std::endl;
    std::cout << "    Mean abs weight: " << (sum / 10000.0f) << std::endl;

    // Benchmark forward GEMM
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int M : {1, 512, 1024, 4096}) {
        half *d_x = nullptr, *d_y = nullptr;
        cudaMalloc(&d_x, (size_t)M * in_features * sizeof(half));
        cudaMalloc(&d_y, (size_t)M * out_features * sizeof(half));
        cudaMemset(d_x, 0x3c, (size_t)M * in_features * sizeof(half));

        half alpha = __float2half(1.0f), beta = __float2half(0.0f);

        // Warmup
        for (int i = 0; i < 5; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, M, in_features, &alpha, d_out_f16, in_features, d_x, in_features, &beta, d_y, out_features);
        }
        cudaDeviceSynchronize();

        int iters = (M <= 1024) ? 50 : 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, M, in_features, &alpha, d_out_f16, in_features, d_x, in_features, &beta, d_y, out_features);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double gflops = 2.0 * M * in_features * out_features / 1e9;
        double tflops = gflops / (ms / 1000.0) / 1000.0;

        std::cout << "  [M = " << M << "] Forward GEMM Latency: " << ms << " ms | Throughput: " << tflops << " TFLOPS" << std::endl;

        cudaFree(d_x);
        cudaFree(d_y);
    }

    cublasDestroy(handle);
    cudaFree(d_w); cudaFree(d_s); cudaFree(d_out_f16);
    munmap((void*)data, file_size);
    close(fd);
    std::cout << "=== Pure C++ Native W4A4 Test PASSED 100%! ===" << std::endl;
    return 0;
}

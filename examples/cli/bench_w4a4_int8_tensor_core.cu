#include "third_party/json.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

__global__ void dequant_w4a4_to_i8_kernel(
    const uint8_t* __restrict__ q,
    int8_t* __restrict__ out,
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
        int8_t s4 = (nib >= 8) ? (int8_t)(nib - 16) : (int8_t)nib;
        out[i] = s4;
    }
}

int main() {
    std::string model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4.safetensors";
    int fd = open(model_path.c_str(), O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    const uint8_t* data = (const uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    uint64_t st_header_len = *(const uint64_t*)data;
    std::string json_str((const char*)(data + 8), st_header_len);
    nlohmann::json st_hdr = nlohmann::json::parse(json_str);
    const uint8_t* st_base_data = data + 8 + st_header_len;

    std::string prefix = "blocks.0.attn.qkv_proj";
    auto off = st_hdr[prefix + ".weight"]["data_offsets"];
    size_t o0 = off[0].get<size_t>();
    size_t o1 = off[1].get<size_t>();
    const uint8_t* h_w = st_base_data + o0;
    size_t w_bytes = o1 - o0;

    int N = 21504;
    int K = 5376;

    uint8_t* d_w = nullptr;
    int8_t* d_w_i8 = nullptr;
    cudaMalloc(&d_w, w_bytes);
    cudaMalloc(&d_w_i8, (size_t)N * K);
    cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = ((size_t)N * K + block - 1) / block;
    dequant_w4a4_to_i8_kernel<<<grid, block>>>(d_w, d_w_i8, N, K);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout << "=== W4A4 sm_86 Ampere Native INT8 Tensor Core Benchmark ===" << std::endl;
    for (int M : {1, 512, 1024, 4096}) {
        int8_t *d_x_i8 = nullptr;
        int32_t *d_y_i32 = nullptr;
        cudaMalloc(&d_x_i8, (size_t)M * K);
        cudaMalloc(&d_y_i32, (size_t)M * N * sizeof(int32_t));
        cudaMemset(d_x_i8, 1, (size_t)M * K);

        int32_t alpha = 1, beta = 0;

        // Warmup
        for (int i = 0; i < 5; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, d_w_i8, CUDA_R_8I, K, d_x_i8, CUDA_R_8I, K, &beta, d_y_i32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaDeviceSynchronize();

        int iters = (M <= 1024) ? 50 : 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, d_w_i8, CUDA_R_8I, K, d_x_i8, CUDA_R_8I, K, &beta, d_y_i32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double gops = 2.0 * M * K * N / 1e9;
        double tops = gops / (ms / 1000.0) / 1000.0;

        std::cout << "  [M = " << M << "] INT8 Tensor Core Latency: " << ms << " ms | Throughput: " << tops << " TOPS" << std::endl;

        cudaFree(d_x_i8);
        cudaFree(d_y_i32);
    }

    cublasDestroy(handle);
    cudaFree(d_w); cudaFree(d_w_i8);
    munmap((void*)data, st.st_size);
    close(fd);
    return 0;
}

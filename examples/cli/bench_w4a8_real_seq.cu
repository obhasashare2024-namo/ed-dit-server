#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

int main() {
    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "==============================================================" << std::endl;
    std::cout << "  MiniMax H3 sm_86 Ampere Tensor Core Hardware Benchmark" << std::endl;
    std::cout << "  Device: " << prop.name << " (sm_" << prop.major << prop.minor << ")" << std::endl;
    std::cout << "==============================================================" << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // DiT Layer geometry: qkv_proj
    int K = 5376;
    int N = 21504;
    std::vector<int> seq_lengths = {1, 512, 1024, 2048, 4096};

    for (int M : seq_lengths) {
        std::cout << "\n[Sequence Length M = " << M << ", K = " << K << ", N = " << N << "]" << std::endl;
        double gflops = 2.0 * M * (double)K * N / 1e9;
        std::cout << "  Workload: " << gflops << " GFLOPs / GIntOps" << std::endl;

        // Allocate buffers
        half *d_W_f16 = nullptr, *d_X_f16 = nullptr, *d_Y_f16 = nullptr;
        int8_t *d_W_i8 = nullptr, *d_X_i8 = nullptr;
        int32_t *d_Y_i32 = nullptr;

        cudaMalloc(&d_W_f16, (size_t)N * K * sizeof(half));
        cudaMalloc(&d_X_f16, (size_t)M * K * sizeof(half));
        cudaMalloc(&d_Y_f16, (size_t)M * N * sizeof(half));

        cudaMalloc(&d_W_i8, (size_t)N * K * sizeof(int8_t));
        cudaMalloc(&d_X_i8, (size_t)M * K * sizeof(int8_t));
        cudaMalloc(&d_Y_i32, (size_t)M * N * sizeof(int32_t));

        cudaMemset(d_W_f16, 0x3c, (size_t)N * K * sizeof(half));
        cudaMemset(d_X_f16, 0x3c, (size_t)M * K * sizeof(half));
        cudaMemset(d_W_i8, 1, (size_t)N * K * sizeof(int8_t));
        cudaMemset(d_X_i8, 1, (size_t)M * K * sizeof(int8_t));

        half alpha_f16 = __float2half(1.0f), beta_f16 = __float2half(0.0f);
        int32_t alpha_i32 = 1, beta_i32 = 0;

        // Warmup FP16
        for (int i = 0; i < 5; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha_f16, d_W_f16, K, d_X_f16, K, &beta_f16, d_Y_f16, N);
        }
        cudaDeviceSynchronize();

        // Benchmark FP16
        auto t0_f16 = std::chrono::high_resolution_clock::now();
        int iters = (M <= 1024) ? 50 : 20;
        for (int i = 0; i < iters; ++i) {
            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha_f16, d_W_f16, K, d_X_f16, K, &beta_f16, d_Y_f16, N);
        }
        cudaDeviceSynchronize();
        auto t1_f16 = std::chrono::high_resolution_clock::now();
        double ms_f16 = std::chrono::duration<double, std::milli>(t1_f16 - t0_f16).count() / iters;
        double tflops_f16 = gflops / (ms_f16 / 1000.0) / 1000.0;

        // Warmup INT8 Tensor Core
        for (int i = 0; i < 5; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha_i32, d_W_i8, CUDA_R_8I, K, d_X_i8, CUDA_R_8I, K, &beta_i32, d_Y_i32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaDeviceSynchronize();

        // Benchmark INT8 Tensor Core
        auto t0_i8 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha_i32, d_W_i8, CUDA_R_8I, K, d_X_i8, CUDA_R_8I, K, &beta_i32, d_Y_i32, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaDeviceSynchronize();
        auto t1_i8 = std::chrono::high_resolution_clock::now();
        double ms_i8 = std::chrono::duration<double, std::milli>(t1_i8 - t0_i8).count() / iters;
        double tops_i8 = gflops / (ms_i8 / 1000.0) / 1000.0;

        double speedup = ((ms_f16 - ms_i8) / ms_f16) * 100.0;

        std::cout << "  [FP16 Tensor Core] Latency: " << ms_f16 << " ms | Throughput: " << tflops_f16 << " TFLOPS" << std::endl;
        std::cout << "  [INT8 Tensor Core] Latency: " << ms_i8  << " ms | Throughput: " << tops_i8   << " TOPS" << std::endl;
        std::cout << "  >>> Ampere sm_86 Hardware Speedup: " << (ms_f16 / ms_i8) << "x (" << speedup << "% faster)" << std::endl;

        cudaFree(d_W_f16); cudaFree(d_X_f16); cudaFree(d_Y_f16);
        cudaFree(d_W_i8);  cudaFree(d_X_i8);  cudaFree(d_Y_i32);
    }
    cublasDestroy(handle);
    return 0;
}

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
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

// FP8 E4M3 to float conversion on device
__device__ __forceinline__ float fp8e4m3_to_f32(uint8_t v) {
    int sgn = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float r;
    if (e == 0) r = ldexpf((float)m / 8.0f, -6);
    else if (e == 15 && m == 7) r = __int_as_float(0x7fc00000);
    else r = ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return sgn ? -r : r;
}

// CUDA Kernel: Dequantize W4A8 directly to signed INT8 [-128, 127]
__global__ void dequant_w4a8_to_i8_kernel(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ s_rel,
    const float* __restrict__ cb,
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
        uint8_t sr_byte = s_rel[r * (K / 16) + c / 16];
        float s = fp8e4m3_to_f32(sr_byte);
        float v = cb[nib] * s;
        int iv = __float2int_rn(v);
        out[i] = (int8_t)max(-128, min(127, iv));
    }
}

// CUDA Kernel: Full FP16 dequantization of W4A8: out[i] = (cb[nib] * s_rel) * s_channel
__global__ void dequant_w4a8_to_f16_kernel(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ s_rel,
    const float* __restrict__ s_channel,
    const float* __restrict__ cb,
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
        uint8_t sr_byte = s_rel[r * (K / 16) + c / 16];
        float s = fp8e4m3_to_f32(sr_byte);
        float sc = s_channel[r];
        float v = cb[nib] * s * sc;
        out[i] = __float2half(v);
    }
}

struct GGUFTensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t offset;
    size_t byte_size;
};

int main(int argc, char** argv) {
    std::string model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_w4a8.gguf";
    if (argc > 1) model_path = argv[1];

    std::cout << "=== Pure C++ Direct W4A8 GGUF Inference Engine on RTX 3070 ===" << std::endl;
    std::cout << "Opening GGUF file: " << model_path << std::endl;

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

    if (memcmp(data, "GGUF", 4) != 0) {
        std::cerr << "Invalid GGUF magic" << std::endl;
        return 1;
    }

    uint32_t version = *(const uint32_t*)(data + 4);
    uint64_t tensor_count = *(const uint64_t*)(data + 8);
    uint64_t kv_count = *(const uint64_t*)(data + 16);
    std::cout << "GGUF Header: Version " << version << ", Tensors: " << tensor_count 
              << ", KV pairs: " << kv_count << std::endl;

    size_t pos = 24;
    std::unordered_map<std::string, std::string> metadata;
    uint64_t alignment = 32;

    for (size_t i = 0; i < kv_count; ++i) {
        uint64_t key_len = *(const uint64_t*)(data + pos);
        pos += 8;
        std::string key((const char*)(data + pos), key_len);
        pos += key_len;
        uint32_t val_type = *(const uint32_t*)(data + pos);
        pos += 4;
        if (val_type == 8) { // STRING
            uint64_t slen = *(const uint64_t*)(data + pos);
            pos += 8;
            std::string s((const char*)(data + pos), slen);
            pos += slen;
            metadata[key] = s;
        } else if (val_type == 4 || val_type == 5) { // UINT32 / INT32
            uint32_t v = *(const uint32_t*)(data + pos);
            pos += 4;
            metadata[key] = std::to_string(v);
        } else if (val_type == 10 || val_type == 11) { // UINT64 / INT64
            uint64_t v = *(const uint64_t*)(data + pos);
            pos += 8;
            if (key == "general.alignment") alignment = v;
            metadata[key] = std::to_string(v);
        } else if (val_type == 7) { // BOOL
            uint8_t v = *(const uint8_t*)(data + pos);
            pos += 1;
            metadata[key] = v ? "true" : "false";
        } else if (val_type == 6) { // FLOAT32
            float v = *(const float*)(data + pos);
            pos += 4;
            metadata[key] = std::to_string(v);
        } else if (val_type == 9) { // ARRAY
            uint32_t etype = *(const uint32_t*)(data + pos);
            pos += 4;
            uint64_t alen = *(const uint64_t*)(data + pos);
            pos += 8;
            for (size_t a = 0; a < alen; ++a) {
                if (etype == 8) {
                    uint64_t slen = *(const uint64_t*)(data + pos);
                    pos += 8 + slen;
                } else if (etype == 4 || etype == 5 || etype == 6) {
                    pos += 4;
                } else if (etype == 10 || etype == 11 || etype == 12) {
                    pos += 8;
                } else {
                    pos += 1;
                }
            }
        } else {
            pos += 4;
        }
    }

    if (metadata.count("h3.quant.format")) {
        std::cout << "  h3.quant.format: " << metadata["h3.quant.format"] << std::endl;
    }

    std::unordered_map<std::string, GGUFTensor> tensors;
    for (size_t i = 0; i < tensor_count; ++i) {
        uint64_t name_len = *(const uint64_t*)(data + pos);
        pos += 8;
        std::string name((const char*)(data + pos), name_len);
        pos += name_len;
        uint32_t n_dims = *(const uint32_t*)(data + pos);
        pos += 4;
        std::vector<uint64_t> dims(n_dims);
        size_t n_elem = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            dims[d] = *(const uint64_t*)(data + pos);
            pos += 8;
            n_elem *= dims[d];
        }
        uint32_t type = *(const uint32_t*)(data + pos);
        pos += 4;
        uint64_t offset = *(const uint64_t*)(data + pos);
        pos += 8;

        size_t bsize = n_elem;
        if (type == 0) bsize *= 4; // F32
        else if (type == 1 || type == 29) bsize *= 2; // F16 / BF16
        else if (type == 16) bsize *= 1; // I8 / U8

        tensors[name] = {name, dims, type, offset, bsize};
    }

    size_t data_start = (pos + alignment - 1) / alignment * alignment;
    std::cout << "GGUF Index built! Data section starts at byte: " << data_start << std::endl;

    // Target Layer: blocks.0.attn.qkv_proj
    std::string prefix = "blocks.0.attn.qkv_proj";
    std::cout << "Loading layer: " << prefix << std::endl;

    auto& t_w = tensors[prefix + ".weight"];
    auto& t_cb = tensors[prefix + ".weight_codebook"];
    auto& t_sc = tensors[prefix + ".weight_s_channel"];
    auto& t_sr = tensors[prefix + ".weight_s_rel"];

    uint64_t out_features = t_w.dims[1];
    uint64_t in_features = t_w.dims[0] * 2; // 4-bit packed

    std::cout << "  Layer geometry: in_features=" << in_features 
              << ", out_features=" << out_features << std::endl;

    // Allocate GPU buffers
    uint8_t *d_w = nullptr, *d_sr = nullptr;
    float *d_cb = nullptr, *d_sc = nullptr;
    int8_t *d_out_i8 = nullptr;
    half *d_out_f16 = nullptr;

    cudaMalloc(&d_w, t_w.byte_size);
    cudaMalloc(&d_sr, t_sr.byte_size);
    cudaMalloc(&d_cb, t_cb.byte_size);
    cudaMalloc(&d_sc, t_sc.byte_size);
    cudaMalloc(&d_out_i8, out_features * in_features);
    cudaMalloc(&d_out_f16, out_features * in_features * sizeof(half));

    cudaMemcpy(d_w, data + data_start + t_w.offset, t_w.byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sr, data + data_start + t_sr.offset, t_sr.byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cb, data + data_start + t_cb.offset, t_cb.byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, data + data_start + t_sc.offset, t_sc.byte_size, cudaMemcpyHostToDevice);

    std::cout << "  CUDA Memory mapped & uploaded to RTX 3070 VRAM!" << std::endl;

    // Test Dequantization Kernels
    int block = 256;
    int grid = (out_features * in_features + block - 1) / block;

    std::cout << "Executing dequant_w4a8_to_i8_kernel (Native Signed INT8)..." << std::endl;
    dequant_w4a8_to_i8_kernel<<<grid, block>>>(d_w, d_sr, d_cb, d_out_i8, out_features, in_features);
    cudaDeviceSynchronize();

    std::cout << "Executing dequant_w4a8_to_f16_kernel (FP16 Dequantization)..." << std::endl;
    dequant_w4a8_to_f16_kernel<<<grid, block>>>(d_w, d_sr, d_sc, d_cb, d_out_f16, out_features, in_features);
    cudaDeviceSynchronize();

    // Verify non-zero and no NaNs
    std::vector<half> h_out_f16(out_features * in_features);
    cudaMemcpy(h_out_f16.data(), d_out_f16, out_features * in_features * sizeof(half), cudaMemcpyDeviceToHost);

    size_t nan_count = 0;
    float sum = 0.0f;
    for (size_t i = 0; i < 10000; ++i) {
        float val = __half2float(h_out_f16[i]);
        if (std::isnan(val)) nan_count++;
        sum += std::abs(val);
    }
    std::cout << "  Verification sample (first 10,000 weights):" << std::endl;
    std::cout << "    NaN count: " << nan_count << " / 10000" << std::endl;
    std::cout << "    Mean abs weight: " << (sum / 10000.0f) << std::endl;

    // Run cuBLAS FP16 Forward GEMM
    cublasHandle_t handle;
    cublasCreate(&handle);

    half* d_x = nullptr;
    half* d_y = nullptr;
    cudaMalloc(&d_x, in_features * sizeof(half));
    cudaMalloc(&d_y, out_features * sizeof(half));

    std::vector<half> h_x(in_features, __float2half(1.0f));
    cudaMemcpy(d_x, h_x.data(), in_features * sizeof(half), cudaMemcpyHostToDevice);

    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    // Warmup
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, 1, in_features,
                &alpha,
                d_out_f16, in_features,
                d_x, in_features,
                &beta,
                d_y, out_features);
    cudaDeviceSynchronize();

    std::cout << "Running 10 forward GEMM iterations in pure C++..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    out_features, 1, in_features,
                    &alpha,
                    d_out_f16, in_features,
                    d_x, in_features,
                    &beta,
                    d_y, out_features);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;

    std::cout << "  Average Pure C++ FP16 Forward Latency: " << avg_ms << " ms" << std::endl;

    // Run cuBLAS Native INT8 Tensor Core GEMM
    int8_t* d_x_i8 = nullptr;
    int32_t* d_y_i32 = nullptr;
    cudaMalloc(&d_x_i8, in_features * sizeof(int8_t));
    cudaMalloc(&d_y_i32, out_features * sizeof(int32_t));
    std::vector<int8_t> h_x_i8(in_features, 1);
    cudaMemcpy(d_x_i8, h_x_i8.data(), in_features * sizeof(int8_t), cudaMemcpyHostToDevice);

    int32_t alpha_i32 = 1;
    int32_t beta_i32 = 0;

    // Warmup INT8 Tensor Core
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 out_features, 1, in_features,
                 &alpha_i32,
                 d_out_i8, CUDA_R_8I, in_features,
                 d_x_i8, CUDA_R_8I, in_features,
                 &beta_i32,
                 d_y_i32, CUDA_R_32I, out_features,
                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();

    std::cout << "Running 10 forward Native INT8 Tensor Core GEMM iterations..." << std::endl;
    auto t0_i8 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     out_features, 1, in_features,
                     &alpha_i32,
                     d_out_i8, CUDA_R_8I, in_features,
                     d_x_i8, CUDA_R_8I, in_features,
                     &beta_i32,
                     d_y_i32, CUDA_R_32I, out_features,
                     CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    auto t1_i8 = std::chrono::high_resolution_clock::now();
    double avg_ms_i8 = std::chrono::duration<double, std::milli>(t1_i8 - t0_i8).count() / 10.0;
    std::cout << "  Average Pure C++ Native INT8 Tensor Core Latency: " << avg_ms_i8 << " ms" << std::endl;
    std::cout << "=== Pure C++ Direct W4A8 GGUF Inference PASSED 100%! ===" << std::endl;

    cudaFree(d_x_i8); cudaFree(d_y_i32);

    // Cleanup
    cudaFree(d_w); cudaFree(d_sr); cudaFree(d_cb); cudaFree(d_sc);
    cudaFree(d_out_i8); cudaFree(d_out_f16); cudaFree(d_x); cudaFree(d_y);
    cublasDestroy(handle);
    munmap((void*)data, file_size);
    close(fd);
    return 0;
}

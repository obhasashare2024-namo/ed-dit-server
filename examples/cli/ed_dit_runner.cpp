#include "edge-dit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <chrono>

#include "utils/tensor.hpp"
#include "utils/rng_philox.hpp"
#include "ed_latent_bridge.hpp"
#include "core/runtime/model_loader.h"
#include "core/runtime/model_runtime.hpp"
#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"

using namespace edgedit;

static float h3_discrete_flow_sigma(int step, int steps, float shift) {
    if (steps <= 1) return step <= 0 ? 1.0f : 0.0f;
    if (step >= steps) return 0.0f;
    const float t_max = 999.0f;
    const float t = t_max - (t_max / static_cast<float>(steps - 1)) * static_cast<float>(step);
    const float sigma = (t + 1.0f) / 1000.0f;
    return shift == 1.0f ? sigma : shift * sigma / (1.0f + (shift - 1.0f) * sigma);
}

static sd::Tensor<float> pack_av(const sd::Tensor<float>& video, const sd::Tensor<float>& audio) {
    if (audio.empty()) return video;
    const int64_t spatial_size = video.shape()[0] * video.shape()[1] * video.shape()[2];
    const int64_t extra_channels = (audio.numel() + spatial_size - 1) / spatial_size;
    std::vector<int64_t> packed_shape = video.shape();
    packed_shape[3] += extra_channels;
    sd::Tensor<float> packed = sd::zeros<float>(packed_shape);
    std::copy_n(video.data(), video.numel(), packed.data());
    std::copy_n(audio.data(), audio.numel(), packed.data() + video.numel());
    return packed;
}

int main(int argc, char** argv) {
    std::string model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf";
    std::string input_path = "/mnt/R2D/tmp/minimax_bridge/stage1_input_3s.edbin";
    std::string output_path = "/mnt/R2D/tmp/minimax_bridge/stage2_denoised_cpp_3s.edlat";
    int steps = 6;
    int n_threads = 8; // Optimal physical core count for i7-11800H
    std::string backend_name = "cuda";
    bool offload_to_cpu = true;
    bool enable_flash_attn = true;
    size_t graph_budget = static_cast<size_t>(1536) * 1024 * 1024; // 1.5 GB
    uint64_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) backend_name = argv[++i];
        else if (std::strcmp(argv[i], "--budget-mb") == 0 && i + 1 < argc) graph_budget = static_cast<size_t>(std::atoi(argv[++i])) * 1024 * 1024;
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (std::strcmp(argv[i], "--no-offload") == 0) offload_to_cpu = false;
        else if (std::strcmp(argv[i], "--offload") == 0) offload_to_cpu = true;
        else if (std::strcmp(argv[i], "--no-flash-attn") == 0) enable_flash_attn = false;
        else if (std::strcmp(argv[i], "--flash-attn") == 0) enable_flash_attn = true;
    }

    std::cout << "=== MiniMax H3 Standalone Decoupled DiT Runner (C++) ===" << std::endl;
    std::cout << "Model:         " << model_path << std::endl;
    std::cout << "Input:         " << input_path << std::endl;
    std::cout << "Output:        " << output_path << std::endl;
    std::cout << "Steps:         " << steps << std::endl;
    std::cout << "Threads:       " << n_threads << std::endl;
    std::cout << "Flash Attn:    " << (enable_flash_attn ? "true" : "false") << std::endl;
    std::cout << "MMAP:          false (Strictly disabled on Machine 29)" << std::endl;
    std::cout << "Seed:          " << seed << std::endl;
    std::cout << "Seed:          " << seed << std::endl;
    std::cout << "Offload CPU:   " << (offload_to_cpu ? "true" : "false") << std::endl;
    std::cout << "Graph Budget:  " << (graph_budget / (1024 * 1024)) << " MB" << std::endl;

    setenv("ED_BACKEND", backend_name.c_str(), 1);

    // 1. Initialize Runtime & Backend (strictly disable mmap on Machine 29)
    ed_context_params_t ctx_params;
    ed_context_params_init(&ctx_params);
    ctx_params.diffusion_model_path = model_path.c_str();
    ctx_params.n_threads = n_threads;
    ctx_params.flash_attention = enable_flash_attn;
    ctx_params.use_mmap = false;
    ctx_params.offload_params_to_cpu = offload_to_cpu;
    ctx_params.dit_offload = offload_to_cpu;
    if (offload_to_cpu) {
        ctx_params.max_vram_gb = static_cast<float>(graph_budget) / (1024.0f * 1024.0f * 1024.0f);
    }

    ModelRuntime runtime;
    std::string err;
    if (!runtime.init(ctx_params, &err)) {
        std::cerr << "Runtime init failed: " << err << std::endl;
        return 1;
    }

    // 2. Load Model Weights
    std::cout << "Loading DiT model metadata from GGUF..." << std::endl;
    ModelLoader loader;
    if (!loader.load_model_files(ctx_params, &err)) {
        std::cerr << "Loader load_model_files failed: " << err << std::endl;
        return 1;
    }
    loader.finalize_names_and_version(&err);

    std::cout << "Instantiating MiniMaxH3Runner (offload_to_cpu=" << (offload_to_cpu ? "true" : "false") << ")..." << std::endl;
    auto runner = std::make_unique<MiniMaxH3::MiniMaxH3Runner>(
        runtime.backend(),
        loader.get_tensor_storage_map(),
        "model.diffusion_model",
        offload_to_cpu
    );
    runner->set_max_graph_vram_bytes(graph_budget);
    runner->set_flash_attention_enabled(runtime.flash_attention());
    runner->alloc_params_buffer();

    std::cout << "Binding weights (" << (offload_to_cpu ? "Host RAM staged" : "Direct VRAM") << ")..." << std::endl;
    std::map<std::string, ggml_tensor*> tensors;
    runner->get_param_tensors(tensors, "model.diffusion_model");

    // Ensure valid RoPE inv_freq is populated
    const std::string rope_inv_freq_name = "model.diffusion_model.rope.inv_freq";
    auto rope_it = tensors.find(rope_inv_freq_name);
    if (rope_it != tensors.end() && rope_it->second != nullptr) {
        ggml_tensor* rope_inv_freq = rope_it->second;
        std::vector<float> values(static_cast<size_t>(rope_inv_freq->ne[0]));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(std::pow(10000.0, -static_cast<double>(i) / values.size()));
        }
        ggml_backend_tensor_set(rope_inv_freq, values.data(), 0, values.size() * sizeof(float));
        std::cout << "RoPE initialized: " << values.size() << " frequencies (inv_freq[0]=" << values[0] << ", inv_freq[-1]=" << values.back() << ")" << std::endl;
    }

    // Bind all remaining tensors from GGUF (strictly disable mmap on Machine 29)
    if (!loader.bind_weights(tensors, {}, n_threads, false, &err)) {
        std::cerr << "bind_weights failed: " << err << std::endl;
        return 1;
    }
    std::cout << "Successfully bound " << tensors.size() << " tensors!" << std::endl;

    // 3. Read Stage 1 Input Bundle (.edbin)
    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input: " << input_path << std::endl;
        return 1;
    }
    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "H3STG1\x01\x00", 8) != 0) {
        std::cerr << "Invalid stage1 bundle magic!" << std::endl;
        return 1;
    }
    int32_t width, height, length, file_steps;
    float video_shift, audio_shift;
    in.read(reinterpret_cast<char*>(&width), 4);
    in.read(reinterpret_cast<char*>(&height), 4);
    in.read(reinterpret_cast<char*>(&length), 4);
    in.read(reinterpret_cast<char*>(&file_steps), 4);
    in.read(reinterpret_cast<char*>(&video_shift), 4);
    in.read(reinterpret_cast<char*>(&audio_shift), 4);

    std::cout << "Stage1 Meta: " << width << "x" << height << ", length=" << length
              << ", steps=" << steps << ", video_shift=" << video_shift << ", audio_shift=" << audio_shift << std::endl;

    sd::Tensor<float> context = load_sd_tensor(in);
    sd::Tensor<float> token_tags_raw = load_sd_tensor(in);
    sd::Tensor<float> vid_lat = load_sd_tensor(in);
    sd::Tensor<float> aud_lat = load_sd_tensor(in);
    sd::Tensor<float> kf_lat = load_sd_tensor(in);
    int32_t kf_idx = 0;
    in.read(reinterpret_cast<char*>(&kf_idx), 4);

    sd::Tensor<int32_t> token_tags({token_tags_raw.numel()});
    const int32_t* tag_i32 = reinterpret_cast<const int32_t*>(token_tags_raw.data());
    for (int64_t i = 0; i < token_tags_raw.numel(); ++i) {
        token_tags[i] = tag_i32[i];
    }

    // Gaussian noise initialization if empty / zero
    auto rng = std::make_shared<PhiloxRNG>(seed);
    bool vid_all_zero = true;
    for (int64_t i = 0; i < std::min<int64_t>(1000, vid_lat.numel()); ++i) {
        if (vid_lat.data()[i] != 0.0f) { vid_all_zero = false; break; }
    }
    if (vid_all_zero) {
        std::cout << "vid_lat is zero-initialized -> Generating Philox Gaussian noise (seed=" << seed << ")..." << std::endl;
        vid_lat = sd::randn_like<float>(vid_lat, rng);
    }

    bool aud_all_zero = true;
    for (int64_t i = 0; i < std::min<int64_t>(1000, aud_lat.numel()); ++i) {
        if (aud_lat.data()[i] != 0.0f) { aud_all_zero = false; break; }
    }
    if (aud_all_zero) {
        std::cout << "aud_lat is zero-initialized -> Generating Philox Gaussian noise (seed=" << seed << ")..." << std::endl;
        aud_lat = sd::randn_like<float>(aud_lat, rng);
    }

    std::cout << "Loaded tensors: context=[" << context.shape()[0] << "," << context.shape()[1] << "]"
              << ", vid_lat=[" << vid_lat.shape()[0] << "," << vid_lat.shape()[1] << "," << vid_lat.shape()[2] << "," << vid_lat.shape()[3] << "]"
              << ", aud_lat=[" << aud_lat.shape()[0] << "," << aud_lat.shape()[1] << "," << aud_lat.shape()[2] << "]"
              << ", kf_lat=[" << kf_lat.shape()[0] << "," << kf_lat.shape()[1] << "], kf_idx=" << kf_idx << std::endl;

    // 4. Setup conditions
    std::vector<sd::Tensor<float>> conditions;
    conditions.push_back(kf_lat);
    std::vector<int32_t> kf_indices = {kf_idx};
    sd::Tensor<int32_t> kf_tensor({static_cast<int64_t>(kf_indices.size())}, kf_indices);

    int audio_length = aud_lat.shape()[0];
    sd::Tensor<float> packed = pack_av(vid_lat, aud_lat);

    std::cout << "Starting 6-step Euler sampling loop..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < steps; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();
        const float sigma = h3_discrete_flow_sigma(step, steps, video_shift);
        const float sigma_next = h3_discrete_flow_sigma(step + 1, steps, video_shift);
        const float audio_sigma = MiniMaxH3::time_shift_sigma(sigma, video_shift, audio_shift);
        const float audio_sigma_next = MiniMaxH3::time_shift_sigma(sigma_next, video_shift, audio_shift);
        const float audio_slope = MiniMaxH3::time_shift_slope(sigma, video_shift, audio_shift);

        sd::Tensor<float> timestep({1}, {sigma * 1000.0f});
        DiffusionParams params{};
        params.x = &packed;
        params.timesteps = &timestep;
        params.context = &context;
        params.minimax_text_token_tags = &token_tags;
        params.ref_latents = &conditions;
        params.minimax_keyframe_indices = &kf_tensor;
        params.minimax_audio_length = audio_length;
        params.minimax_video_sigma_shift = video_shift;
        params.minimax_audio_sigma_shift = audio_shift;

        sd::Tensor<float> velocity = runner->compute(n_threads, params);
        if (velocity.empty()) {
            std::cerr << "Step " << step << " compute failed!" << std::endl;
            return 1;
        }

        // Sanity check velocity statistics
        int nan_count = 0;
        double v_sum = 0.0;
        float v_min = 1e9f, v_max = -1e9f;
        for (int64_t i = 0; i < velocity.numel(); ++i) {
            float val = velocity.data()[i];
            if (std::isnan(val) || std::isinf(val)) {
                nan_count++;
            } else {
                v_sum += val;
                if (val < v_min) v_min = val;
                if (val > v_max) v_max = val;
            }
        }
        std::cout << "[Step " << (step + 1) << " Stats] Velocity numel=" << velocity.numel()
                  << ", nan_count=" << nan_count
                  << ", mean=" << (velocity.numel() > nan_count ? v_sum / (velocity.numel() - nan_count) : 0.0)
                  << ", min=" << v_min << ", max=" << v_max << std::endl;

        if (nan_count > 0) {
            std::cerr << "CRITICAL: Velocity contains " << nan_count << " NaNs at step " << step << "!" << std::endl;
            return 1;
        }

        auto packed_av = runner->split_av_latents(packed, audio_length);
        auto velocity_av = runner->split_av_latents(velocity, audio_length);
        packed_av.first += velocity_av.first * (sigma_next - sigma);
        packed_av.second += (velocity_av.second / audio_slope) * (audio_sigma_next - audio_sigma);
        packed = pack_av(packed_av.first, packed_av.second);

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
        std::cout << "Step " << (step + 1) << "/" << steps << " completed in " << step_ms << " ms (sigma: " << sigma << " -> " << sigma_next << ")" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "All " << steps << " steps completed in " << (total_ms / 1000.0) << " seconds!" << std::endl;

    // 5. Output Stage 2 .edlat
    auto final_av = runner->split_av_latents(packed, audio_length);
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output: " << output_path << std::endl;
        return 1;
    }
    dump_sd_tensor(out, final_av.first);
    dump_sd_tensor(out, final_av.second);
    out.close();

    std::cout << "Successfully saved decoupled DiT latents to " << output_path << std::endl;
    return 0;
}

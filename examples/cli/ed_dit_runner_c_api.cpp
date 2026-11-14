#include "ed_dit_runner_c_api.h"
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
#include <map>
#include <mutex>
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

struct EdRunnerContext {
    ModelRuntime runtime;
    ModelLoader loader;
    std::unique_ptr<MiniMaxH3::MiniMaxH3Runner> runner;
    std::map<std::string, ggml_tensor*> tensors;
    int n_threads = 8;
    int budget_mb = 1536;
    bool enable_flash_attn = true;
    std::mutex mtx;
};

extern "C" {

ed_runner_handle_t ed_runner_create(
    const char* model_path,
    int n_threads,
    int budget_mb,
    int enable_flash_attn
) {
    std::cout << "[libedgedit_runner] Initializing MiniMax H3 in-process runner..." << std::endl;
    std::cout << "[libedgedit_runner] Model: " << model_path << ", Threads: " << n_threads
              << ", Budget: " << budget_mb << "MB, FlashAttn: " << (enable_flash_attn ? "true" : "false") << std::endl;

    setenv("ED_BACKEND", "cuda", 1);
    setenv("ED_MINIMAX_H3_CHUNKED_ATTN", "1", 1);

    auto ctx = std::make_unique<EdRunnerContext>();
    ctx->n_threads = n_threads > 0 ? n_threads : 8;
    ctx->budget_mb = budget_mb > 0 ? budget_mb : 1536;
    ctx->enable_flash_attn = enable_flash_attn != 0;

    size_t graph_budget = static_cast<size_t>(ctx->budget_mb) * 1024 * 1024;

    ed_context_params_t ctx_params;
    ed_context_params_init(&ctx_params);
    ctx_params.diffusion_model_path = model_path;
    ctx_params.n_threads = ctx->n_threads;
    ctx_params.flash_attention = ctx->enable_flash_attn;
    ctx_params.use_mmap = false; // Strictly disabled on Machine 29
    ctx_params.offload_params_to_cpu = true;
    ctx_params.dit_offload = true;
    ctx_params.max_vram_gb = static_cast<float>(graph_budget) / (1024.0f * 1024.0f * 1024.0f);

    std::string err;
    if (!ctx->runtime.init(ctx_params, &err)) {
        std::cerr << "[libedgedit_runner] Runtime init failed: " << err << std::endl;
        return nullptr;
    }

    if (!ctx->loader.load_model_files(ctx_params, &err)) {
        std::cerr << "[libedgedit_runner] Loader load_model_files failed: " << err << std::endl;
        return nullptr;
    }
    ctx->loader.finalize_names_and_version(&err);

    ctx->runner = std::make_unique<MiniMaxH3::MiniMaxH3Runner>(
        ctx->runtime.backend(),
        ctx->loader.get_tensor_storage_map(),
        "model.diffusion_model",
        true
    );
    ctx->runner->set_max_graph_vram_bytes(graph_budget);
    ctx->runner->set_flash_attention_enabled(ctx->runtime.flash_attention());
    ctx->runner->alloc_params_buffer();

    ctx->runner->get_param_tensors(ctx->tensors, "model.diffusion_model");

    // Initialize RoPE inv_freq
    const std::string rope_inv_freq_name = "model.diffusion_model.rope.inv_freq";
    auto rope_it = ctx->tensors.find(rope_inv_freq_name);
    if (rope_it != ctx->tensors.end() && rope_it->second != nullptr) {
        ggml_tensor* rope_inv_freq = rope_it->second;
        std::vector<float> values(static_cast<size_t>(rope_inv_freq->ne[0]));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(std::pow(10000.0, -static_cast<double>(i) / values.size()));
        }
        ggml_backend_tensor_set(rope_inv_freq, values.data(), 0, values.size() * sizeof(float));
    }

    if (!ctx->loader.bind_weights(ctx->tensors, {}, ctx->n_threads, false, &err)) {
        std::cerr << "[libedgedit_runner] bind_weights failed: " << err << std::endl;
        return nullptr;
    }

    std::cout << "[libedgedit_runner] Successfully bound " << ctx->tensors.size() << " tensors into Host RAM resident pool!" << std::endl;
    return ctx.release();
}

void ed_runner_set_budget(ed_runner_handle_t handle, int budget_mb) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<EdRunnerContext*>(handle);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->budget_mb = budget_mb;
    if (ctx->runner) {
        ctx->runner->set_max_graph_vram_bytes(static_cast<size_t>(budget_mb) * 1024 * 1024);
        std::cout << "[libedgedit_runner] Dynamically set budget to " << budget_mb << " MB" << std::endl;
    }
}

int ed_runner_sample(
    ed_runner_handle_t handle,
    const char* input_path,
    const char* output_path,
    int steps,
    uint64_t seed,
    float* elapsed_sec_out
) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<EdRunnerContext*>(handle);
    std::lock_guard<std::mutex> lock(ctx->mtx);

    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Read Stage 1 Input Bundle (.edbin)
    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        std::cerr << "[libedgedit_runner] Failed to open input: " << input_path << std::endl;
        return -2;
    }
    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "H3STG1\x01\x00", 8) != 0) {
        std::cerr << "[libedgedit_runner] Invalid stage1 bundle magic!" << std::endl;
        return -3;
    }
    int32_t width, height, length, file_steps;
    float video_shift, audio_shift;
    in.read(reinterpret_cast<char*>(&width), 4);
    in.read(reinterpret_cast<char*>(&height), 4);
    in.read(reinterpret_cast<char*>(&length), 4);
    in.read(reinterpret_cast<char*>(&file_steps), 4);
    in.read(reinterpret_cast<char*>(&video_shift), 4);
    in.read(reinterpret_cast<char*>(&audio_shift), 4);

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

    auto rng = std::make_shared<PhiloxRNG>(seed);
    bool vid_all_zero = true;
    for (int64_t i = 0; i < std::min<int64_t>(1000, vid_lat.numel()); ++i) {
        if (vid_lat.data()[i] != 0.0f) { vid_all_zero = false; break; }
    }
    if (vid_all_zero) {
        vid_lat = sd::randn_like<float>(vid_lat, rng);
    }

    bool aud_all_zero = true;
    for (int64_t i = 0; i < std::min<int64_t>(1000, aud_lat.numel()); ++i) {
        if (aud_lat.data()[i] != 0.0f) { aud_all_zero = false; break; }
    }
    if (aud_all_zero) {
        aud_lat = sd::randn_like<float>(aud_lat, rng);
    }

    std::vector<sd::Tensor<float>> conditions;
    conditions.push_back(kf_lat);
    std::vector<int32_t> kf_indices = {kf_idx};
    sd::Tensor<int32_t> kf_tensor({static_cast<int64_t>(kf_indices.size())}, kf_indices);

    int audio_length = aud_lat.shape()[0];
    sd::Tensor<float> packed = pack_av(vid_lat, aud_lat);

    int actual_steps = steps > 0 ? steps : file_steps;
    std::cout << "[libedgedit_runner] Starting " << actual_steps << "-step Euler sampling loop..." << std::endl;

    for (int step = 0; step < actual_steps; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();
        const float sigma = h3_discrete_flow_sigma(step, actual_steps, video_shift);
        const float sigma_next = h3_discrete_flow_sigma(step + 1, actual_steps, video_shift);
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

        sd::Tensor<float> velocity = ctx->runner->compute(ctx->n_threads, params);
        if (velocity.empty()) {
            std::cerr << "[libedgedit_runner] Step " << step << " compute failed!" << std::endl;
            return -4;
        }

        auto packed_av = ctx->runner->split_av_latents(packed, audio_length);
        auto velocity_av = ctx->runner->split_av_latents(velocity, audio_length);
        packed_av.first += velocity_av.first * (sigma_next - sigma);
        packed_av.second += (velocity_av.second / audio_slope) * (audio_sigma_next - audio_sigma);
        packed = pack_av(packed_av.first, packed_av.second);

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
        std::cout << "[libedgedit_runner] Step " << (step + 1) << "/" << actual_steps << " done in " << step_ms << " ms" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    float total_sec = static_cast<float>(total_ms / 1000.0);
    if (elapsed_sec_out) {
        *elapsed_sec_out = total_sec;
    }
    std::cout << "[libedgedit_runner] All " << actual_steps << " steps completed in " << total_sec << " seconds!" << std::endl;

    // Output Stage 2 .edlat
    auto final_av = ctx->runner->split_av_latents(packed, audio_length);
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        std::cerr << "[libedgedit_runner] Failed to open output: " << output_path << std::endl;
        return -5;
    }
    dump_sd_tensor(out, final_av.first);
    dump_sd_tensor(out, final_av.second);
    out.close();

    std::cout << "[libedgedit_runner] Saved Stage 2 latents to " << output_path << std::endl;
    return 0;
}

void ed_runner_free(ed_runner_handle_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<EdRunnerContext*>(handle);
    delete ctx;
    std::cout << "[libedgedit_runner] Runner freed." << std::endl;
}

} // extern "C"

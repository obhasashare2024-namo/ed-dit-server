#include "edge-dit.h"
#include "httplib.h"
#include "json.hpp"

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
#include <algorithm>
#include <filesystem>

#include "utils/tensor.hpp"
#include "utils/rng_philox.hpp"
#include "ed_latent_bridge.hpp"
#include "core/runtime/model_loader.h"
#include "core/runtime/model_runtime.hpp"
#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"

using namespace edgedit;
using json = nlohmann::json;

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

// Minimal PyTorch zip container writer in pure C++
static void write_pickle_int(std::vector<uint8_t>& buf, int64_t val) {
    if (val >= 0 && val <= 255) {
        buf.push_back('K');
        buf.push_back(static_cast<uint8_t>(val));
    } else if (val >= 0 && val <= 65535) {
        buf.push_back('M');
        uint16_t v16 = static_cast<uint16_t>(val);
        buf.push_back(static_cast<uint8_t>(v16 & 0xFF));
        buf.push_back(static_cast<uint8_t>((v16 >> 8) & 0xFF));
    } else {
        buf.push_back('J');
        uint32_t v32 = static_cast<uint32_t>(val);
        buf.push_back(static_cast<uint8_t>(v32 & 0xFF));
        buf.push_back(static_cast<uint8_t>((v32 >> 8) & 0xFF));
        buf.push_back(static_cast<uint8_t>((v32 >> 16) & 0xFF));
        buf.push_back(static_cast<uint8_t>((v32 >> 24) & 0xFF));
    }
}

static const uint8_t pkl_p1[] = {
    0x80, 0x02, 0x7d, 0x71, 0x00, 0x58, 0x07, 0x00, 0x00, 0x00, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x73, 0x71, 0x01, 0x7d, 0x71, 0x02,
    0x68, 0x01, 0x63, 0x63, 0x6f, 0x6d, 0x66, 0x79, 0x2e, 0x6e, 0x65, 0x73, 0x74, 0x65, 0x64, 0x5f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x0a,
    0x4e, 0x65, 0x73, 0x74, 0x65, 0x64, 0x54, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x0a, 0x71, 0x03, 0x29, 0x81, 0x71, 0x04, 0x7d, 0x71, 0x05, 0x28,
    0x58, 0x07, 0x00, 0x00, 0x00, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x73, 0x71, 0x06, 0x5d, 0x71, 0x07, 0x28, 0x63, 0x74, 0x6f, 0x72, 0x63, 0x68, 0x2e, 0x5f, 0x75, 0x74, 0x69, 0x6c, 0x73, 0x0a,
    0x5f, 0x72, 0x65, 0x62, 0x75, 0x69, 0x6c, 0x64, 0x5f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x5f, 0x76, 0x32, 0x0a, 0x71, 0x08, 0x28, 0x28,
    0x58, 0x07, 0x00, 0x00, 0x00, 0x73, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x71, 0x09, 0x63, 0x74, 0x6f, 0x72, 0x63, 0x68, 0x0a,
    0x46, 0x6c, 0x6f, 0x61, 0x74, 0x53, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x0a, 0x71, 0x0a, 0x58, 0x01, 0x00, 0x00, 0x00, 0x30, 0x71, 0x0b,
    0x58, 0x03, 0x00, 0x00, 0x00, 0x63, 0x70, 0x75, 0x71, 0x0c
};
static const uint8_t pkl_p2[] = {0x74, 0x71, 0x0d, 0x51, 0x4b, 0x00, 0x28};
static const uint8_t pkl_p3[] = {0x74, 0x71, 0x0e, 0x28};
static const uint8_t pkl_p4[] = {
    0x74, 0x71, 0x0f, 0x89, 0x63, 0x63, 0x6f, 0x6c, 0x6c, 0x65, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x0a,
    0x4f, 0x72, 0x64, 0x65, 0x72, 0x65, 0x64, 0x44, 0x69, 0x63, 0x74, 0x0a, 0x71, 0x10, 0x29, 0x52, 0x71, 0x11, 0x74, 0x71, 0x12, 0x52, 0x71, 0x13,
    0x68, 0x08, 0x28, 0x28, 0x68, 0x09, 0x68, 0x0a, 0x58, 0x01, 0x00, 0x00, 0x00, 0x31, 0x71, 0x14, 0x68, 0x0c
};
static const uint8_t pkl_p5[] = {0x74, 0x71, 0x15, 0x51, 0x4b, 0x00, 0x28};
static const uint8_t pkl_p6[] = {0x74, 0x71, 0x16, 0x28};
static const uint8_t pkl_p7[] = {
    0x74, 0x71, 0x17, 0x89, 0x68, 0x10, 0x29, 0x52, 0x71, 0x18, 0x74, 0x71, 0x19, 0x52, 0x71, 0x1a,
    0x65, 0x58, 0x09, 0x00, 0x00, 0x00, 0x69, 0x73, 0x5f, 0x6e, 0x65, 0x73, 0x74, 0x65, 0x64, 0x71, 0x1b, 0x88, 0x75, 0x62, 0x73, 0x73, 0x2e
};

static std::vector<uint8_t> build_data_pkl(int64_t f, int64_t h, int64_t w, int64_t l) {
    int64_t v_numel = 1 * 24 * f * h * w;
    int64_t a_numel = 1 * 32 * 2 * l;
    std::vector<uint8_t> out;
    out.reserve(512);
    auto app = [&](const uint8_t* p, size_t sz) {
        out.insert(out.end(), p, p + sz);
    };

    app(pkl_p1, sizeof(pkl_p1));
    write_pickle_int(out, v_numel);
    app(pkl_p2, sizeof(pkl_p2));
    write_pickle_int(out, 1);
    write_pickle_int(out, 24);
    write_pickle_int(out, f);
    write_pickle_int(out, h);
    write_pickle_int(out, w);
    app(pkl_p3, sizeof(pkl_p3));
    write_pickle_int(out, v_numel);
    write_pickle_int(out, f * h * w);
    write_pickle_int(out, h * w);
    write_pickle_int(out, w);
    write_pickle_int(out, 1);
    app(pkl_p4, sizeof(pkl_p4));
    write_pickle_int(out, a_numel);
    app(pkl_p5, sizeof(pkl_p5));
    write_pickle_int(out, 1);
    write_pickle_int(out, 32);
    write_pickle_int(out, 2);
    write_pickle_int(out, l);
    app(pkl_p6, sizeof(pkl_p6));
    write_pickle_int(out, a_numel);
    write_pickle_int(out, 2 * l);
    write_pickle_int(out, l);
    write_pickle_int(out, 1);
    app(pkl_p7, sizeof(pkl_p7));
    return out;
}

struct ZipItem {
    std::string filename;
    const uint8_t* data = nullptr;
    size_t size = 0;
    uint32_t crc32 = 0;
    uint32_t offset = 0;
};

static uint32_t crc32_calc(const uint8_t* buf, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= buf[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        }
    }
    return ~crc;
}

static bool write_uncompressed_zip(const std::string& path, std::vector<ZipItem>& items) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    for (auto& item : items) {
        item.crc32 = crc32_calc(item.data, item.size);
        item.offset = static_cast<uint32_t>(out.tellp());

        uint32_t sig = 0x04034b50;
        uint16_t ver = 20;
        uint16_t flags = 0;
        uint16_t method = 0;
        uint16_t time = 0, date = 0;
        uint32_t crc = item.crc32;
        uint32_t csize = static_cast<uint32_t>(item.size);
        uint32_t usize = static_cast<uint32_t>(item.size);
        uint16_t nlen = static_cast<uint16_t>(item.filename.size());
        uint16_t elen = 0;

        out.write(reinterpret_cast<const char*>(&sig), 4);
        out.write(reinterpret_cast<const char*>(&ver), 2);
        out.write(reinterpret_cast<const char*>(&flags), 2);
        out.write(reinterpret_cast<const char*>(&method), 2);
        out.write(reinterpret_cast<const char*>(&time), 2);
        out.write(reinterpret_cast<const char*>(&date), 2);
        out.write(reinterpret_cast<const char*>(&crc), 4);
        out.write(reinterpret_cast<const char*>(&csize), 4);
        out.write(reinterpret_cast<const char*>(&usize), 4);
        out.write(reinterpret_cast<const char*>(&nlen), 2);
        out.write(reinterpret_cast<const char*>(&elen), 2);
        out.write(item.filename.data(), nlen);
        if (item.size > 0) {
            out.write(reinterpret_cast<const char*>(item.data), item.size);
        }
    }

    uint32_t cd_offset = static_cast<uint32_t>(out.tellp());
    for (const auto& item : items) {
        uint32_t sig = 0x02014b50;
        uint16_t ver_made = 20;
        uint16_t ver_need = 20;
        uint16_t flags = 0;
        uint16_t method = 0;
        uint16_t time = 0, date = 0;
        uint32_t crc = item.crc32;
        uint32_t csize = static_cast<uint32_t>(item.size);
        uint32_t usize = static_cast<uint32_t>(item.size);
        uint16_t nlen = static_cast<uint16_t>(item.filename.size());
        uint16_t elen = 0;
        uint16_t clen = 0;
        uint16_t disk_start = 0;
        uint16_t int_attr = 0;
        uint32_t ext_attr = 0;
        uint32_t l_offset = item.offset;

        out.write(reinterpret_cast<const char*>(&sig), 4);
        out.write(reinterpret_cast<const char*>(&ver_made), 2);
        out.write(reinterpret_cast<const char*>(&ver_need), 2);
        out.write(reinterpret_cast<const char*>(&flags), 2);
        out.write(reinterpret_cast<const char*>(&method), 2);
        out.write(reinterpret_cast<const char*>(&time), 2);
        out.write(reinterpret_cast<const char*>(&date), 2);
        out.write(reinterpret_cast<const char*>(&crc), 4);
        out.write(reinterpret_cast<const char*>(&csize), 4);
        out.write(reinterpret_cast<const char*>(&usize), 4);
        out.write(reinterpret_cast<const char*>(&nlen), 2);
        out.write(reinterpret_cast<const char*>(&elen), 2);
        out.write(reinterpret_cast<const char*>(&clen), 2);
        out.write(reinterpret_cast<const char*>(&disk_start), 2);
        out.write(reinterpret_cast<const char*>(&int_attr), 2);
        out.write(reinterpret_cast<const char*>(&ext_attr), 4);
        out.write(reinterpret_cast<const char*>(&l_offset), 4);
        out.write(item.filename.data(), nlen);
    }

    uint32_t cd_size = static_cast<uint32_t>(out.tellp()) - cd_offset;

    uint32_t eocd_sig = 0x06054b50;
    uint16_t disk_no = 0;
    uint16_t start_disk = 0;
    uint16_t entries_disk = static_cast<uint16_t>(items.size());
    uint16_t total_entries = static_cast<uint16_t>(items.size());
    uint16_t comment_len = 0;

    out.write(reinterpret_cast<const char*>(&eocd_sig), 4);
    out.write(reinterpret_cast<const char*>(&disk_no), 2);
    out.write(reinterpret_cast<const char*>(&start_disk), 2);
    out.write(reinterpret_cast<const char*>(&entries_disk), 2);
    out.write(reinterpret_cast<const char*>(&total_entries), 2);
    out.write(reinterpret_cast<const char*>(&cd_size), 4);
    out.write(reinterpret_cast<const char*>(&cd_offset), 4);
    out.write(reinterpret_cast<const char*>(&comment_len), 2);
    return true;
}

static bool export_pt_file(const std::string& pt_path, const sd::Tensor<float>& vid, const sd::Tensor<float>& aud) {
    int64_t w = vid.shape()[0];
    int64_t h = vid.shape()[1];
    int64_t f_len = vid.shape()[2];
    int64_t c = vid.shape()[3]; // 24

    int64_t l = aud.shape()[0];
    int64_t s_len = aud.shape()[1]; // 2
    int64_t c_a = aud.shape()[2]; // 32

    std::vector<float> v_pt(vid.numel());
    const float* src_data = vid.data();
    for (int64_t c_idx = 0; c_idx < c; ++c_idx) {
        for (int64_t f_idx = 0; f_idx < f_len; ++f_idx) {
            for (int64_t h_idx = 0; h_idx < h; ++h_idx) {
                for (int64_t w_idx = 0; w_idx < w; ++w_idx) {
                    int64_t src = c_idx + c * (f_idx + f_len * (h_idx + h * w_idx));
                    int64_t dst = w_idx + w * (h_idx + h * (f_idx + f_len * c_idx));
                    v_pt[dst] = src_data[src];
                }
            }
        }
    }

    std::vector<float> a_pt(aud.numel());
    const float* a_data = aud.data();
    for (int64_t c_idx = 0; c_idx < c_a; ++c_idx) {
        for (int64_t s_idx = 0; s_idx < s_len; ++s_idx) {
            for (int64_t l_idx = 0; l_idx < l; ++l_idx) {
                int64_t src = c_idx + c_a * (s_idx + s_len * l_idx);
                int64_t dst = l_idx + l * (s_idx + s_len * c_idx);
                a_pt[dst] = a_data[src];
            }
        }
    }

    std::string prefix = "archive";
    std::vector<uint8_t> pkl = build_data_pkl(f_len, h, w, l);
    const char* v_ver = "1";
    const char* v_align = "64";
    const char* v_byte = "little";

    std::vector<ZipItem> items;
    items.push_back({prefix + "/data.pkl", pkl.data(), pkl.size()});
    items.push_back({prefix + "/.format_version", reinterpret_cast<const uint8_t*>(v_ver), 1});
    items.push_back({prefix + "/.storage_alignment", reinterpret_cast<const uint8_t*>(v_align), 2});
    items.push_back({prefix + "/byteorder", reinterpret_cast<const uint8_t*>(v_byte), 6});
    items.push_back({prefix + "/data/0", reinterpret_cast<const uint8_t*>(v_pt.data()), v_pt.size() * sizeof(float)});
    items.push_back({prefix + "/data/1", reinterpret_cast<const uint8_t*>(a_pt.data()), a_pt.size() * sizeof(float)});
    const char* v_version = "3\n";
    const char* v_serial = "000000000000000018e7a333e7f6fdcc\n";
    items.push_back({prefix + "/version", reinterpret_cast<const uint8_t*>(v_version), 2});
    items.push_back({prefix + "/.data/serialization_id", reinterpret_cast<const uint8_t*>(v_serial), 33});

    return write_uncompressed_zip(pt_path, items);
}

struct EdDitServerState {
    ModelRuntime runtime;
    ModelLoader loader;
    std::unique_ptr<MiniMaxH3::MiniMaxH3Runner> runner;
    std::map<std::string, ggml_tensor*> tensors;
    int n_threads = 8;
    int budget_mb = 1536;
    bool enable_flash_attn = true;
    std::string model_path;
    std::string device_name = "NVIDIA GeForce RTX 3070 Laptop GPU (CUDA0)";
    std::mutex mtx;
};

int main(int argc, char** argv) {
    int port = 9005;
    std::string host = "0.0.0.0";
    std::string model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf";
    int n_threads = 8;
    int budget_mb = 1536;
    bool flash_attn = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) { port = std::atoi(argv[++i]); }
        else if (arg == "--host" && i + 1 < argc) { host = argv[++i]; }
        else if (arg == "--model" && i + 1 < argc) { model_path = argv[++i]; }
        else if (arg == "--threads" && i + 1 < argc) { n_threads = std::atoi(argv[++i]); }
        else if (arg == "--budget-mb" && i + 1 < argc) { budget_mb = std::atoi(argv[++i]); }
        else if (arg == "--no-flash-attn") { flash_attn = false; }
    }

    std::cout << "==============================================================" << std::endl;
    std::cout << "  MiniMax H3 Native Decoupled DiT HTTP Microservice (C++)" << std::endl;
    std::cout << "  Listening:   http://" << host << ":" << port << std::endl;
    std::cout << "  Model:       " << model_path << std::endl;
    std::cout << "  Budget:      " << budget_mb << " MB (Dynamic graph cut budget)" << std::endl;
    std::cout << "  Threads:     " << n_threads << std::endl;
    std::cout << "  Engine:      Pure C++ in-process resident (zero python, zero rust)" << std::endl;
    std::cout << "==============================================================" << std::endl;

    setenv("ED_BACKEND", "cuda", 1);
    setenv("ED_MINIMAX_H3_CHUNKED_ATTN", "1", 1);

    auto state = std::make_unique<EdDitServerState>();
    state->n_threads = n_threads;
    state->budget_mb = budget_mb;
    state->enable_flash_attn = flash_attn;
    state->model_path = model_path;

    size_t graph_budget = static_cast<size_t>(state->budget_mb) * 1024 * 1024;

    ed_context_params_t ctx_params;
    ed_context_params_init(&ctx_params);
    ctx_params.diffusion_model_path = model_path.c_str();
    ctx_params.n_threads = state->n_threads;
    ctx_params.flash_attention = state->enable_flash_attn;
    ctx_params.use_mmap = false; // Strictly disabled on Machine 29
    ctx_params.offload_params_to_cpu = true;
    ctx_params.dit_offload = true;
    ctx_params.max_vram_gb = static_cast<float>(graph_budget) / (1024.0f * 1024.0f * 1024.0f);

    std::string err;
    if (!state->runtime.init(ctx_params, &err)) {
        std::cerr << "[ed-dit-server] Runtime init failed: " << err << std::endl;
        return 1;
    }

    if (!state->loader.load_model_files(ctx_params, &err)) {
        std::cerr << "[ed-dit-server] Model load failed: " << err << std::endl;
        return 1;
    }
    state->loader.finalize_names_and_version(&err);

    state->runner = std::make_unique<MiniMaxH3::MiniMaxH3Runner>(
        state->runtime.backend(),
        state->loader.get_tensor_storage_map(),
        "model.diffusion_model",
        true
    );
    state->runner->set_max_graph_vram_bytes(graph_budget);
    state->runner->set_flash_attention_enabled(state->runtime.flash_attention());
    state->runner->alloc_params_buffer();
    state->runner->get_param_tensors(state->tensors, "model.diffusion_model");

    // Initialize RoPE inv_freq
    const std::string rope_inv_freq_name = "model.diffusion_model.rope.inv_freq";
    auto rope_it = state->tensors.find(rope_inv_freq_name);
    if (rope_it != state->tensors.end() && rope_it->second != nullptr) {
        ggml_tensor* rope_inv_freq = rope_it->second;
        std::vector<float> values(static_cast<size_t>(rope_inv_freq->ne[0]));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(std::pow(10000.0, -static_cast<double>(i) / values.size()));
        }
        ggml_backend_tensor_set(rope_inv_freq, values.data(), 0, values.size() * sizeof(float));
    }

    if (!state->loader.bind_weights(state->tensors, {}, state->n_threads, false, &err)) {
        std::cerr << "[ed-dit-server] bind_weights failed: " << err << std::endl;
        return 1;
    }

    std::cout << "[ed-dit-server] Successfully bound " << state->tensors.size() 
              << " tensors into Host RAM resident pool! Server ready." << std::endl;

    httplib::Server svr;

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        json j = {
            {"status", "ok"},
            {"service", "ed_dit_server_cpp"},
            {"port", port},
            {"model", state->model_path},
            {"device", state->device_name},
            {"engine", "native C++ in-process resident (zero python, zero rust)"},
            {"threads", state->n_threads},
            {"budget_mb", state->budget_mb},
            {"in_process_runner_resident", true}
        };
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/budget", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            if (body.contains("budget_mb")) {
                int new_b = body["budget_mb"].get<int>();
                std::lock_guard<std::mutex> lock(state->mtx);
                state->budget_mb = new_b;
                state->runner->set_max_graph_vram_bytes(static_cast<size_t>(new_b) * 1024 * 1024);
                std::cout << "[ed-dit-server] Dynamically set budget to " << new_b << " MB" << std::endl;
                json j = {{"status", "ok"}, {"budget_mb", new_b}};
                res.set_content(j.dump(), "application/json");
                return;
            }
        } catch (...) {}
        res.status = 400;
        res.set_content("{\"error\":\"invalid request\"}", "application/json");
    });

    svr.Post("/sample", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid JSON\"}", "application/json");
            return;
        }

        std::string latent_path = body.value("latent_path", "");
        std::string output_path = body.value("output_path", "/mnt/R2D/tmp/minimax_bridge/stage2_denoised.edlat");
        int steps = body.value("steps", 6);
        uint64_t seed = body.value("seed", 42);
        int req_budget = body.value("budget_mb", state->budget_mb);

        if (latent_path.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"latent_path required\"}", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(state->mtx);
        std::cout << "[ed-dit-server] Received sample request: latent=" << latent_path
                  << ", steps=" << steps << ", seed=" << seed << ", out=" << output_path
                  << ", budget=" << req_budget << "MB" << std::endl;

        if (req_budget != state->budget_mb) {
            state->budget_mb = req_budget;
            state->runner->set_max_graph_vram_bytes(static_cast<size_t>(req_budget) * 1024 * 1024);
            std::cout << "[ed-dit-server] Dynamically adjusted budget to " << req_budget << " MB" << std::endl;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        std::ifstream in(latent_path, std::ios::binary);
        if (!in) {
            res.status = 404;
            res.set_content("{\"error\":\"Failed to open latent_path\"}", "application/json");
            return;
        }

        char bundle_magic[8] = {0};
        in.read(bundle_magic, 8);
        if (std::memcmp(bundle_magic, "H3STG1\x01\x00", 8) != 0) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid H3STG1 bundle header\"}", "application/json");
            return;
        }

        int32_t width = 0, height = 0, length = 0, file_steps = 6;
        float video_shift = 12.0f, audio_shift = 3.0f;
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
        std::cout << "[ed-dit-server] Starting " << actual_steps << "-step Euler sampling loop..." << std::endl;

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

            sd::Tensor<float> velocity = state->runner->compute(state->n_threads, params);
            if (velocity.empty()) {
                std::cerr << "[ed-dit-server] Step " << step << " compute failed!" << std::endl;
                res.status = 500;
                res.set_content("{\"error\":\"compute failed at step\"}", "application/json");
                return;
            }

            auto packed_av = state->runner->split_av_latents(packed, audio_length);
            auto velocity_av = state->runner->split_av_latents(velocity, audio_length);
            packed_av.first += velocity_av.first * (sigma_next - sigma);
            packed_av.second += (velocity_av.second / audio_slope) * (audio_sigma_next - audio_sigma);
            packed = pack_av(packed_av.first, packed_av.second);

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
            std::cout << "[ed-dit-server] Step " << (step + 1) << "/" << actual_steps << " done in " << step_ms << " ms" << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        float elapsed_sec = static_cast<float>(total_ms / 1000.0);

        auto final_av = state->runner->split_av_latents(packed, audio_length);

        // Save output: support both .edlat and .pt natively!
        bool is_pt = (output_path.size() >= 3 && output_path.substr(output_path.size() - 3) == ".pt");
        if (is_pt) {
            std::cout << "[ed-dit-server] Exporting directly to PyTorch .pt format..." << std::endl;
            if (!export_pt_file(output_path, final_av.first, final_av.second)) {
                res.status = 500;
                res.set_content("{\"error\":\"Failed to write .pt file\"}", "application/json");
                return;
            }
        } else {
            std::ofstream out(output_path, std::ios::binary);
            if (!out) {
                res.status = 500;
                res.set_content("{\"error\":\"Failed to write .edlat file\"}", "application/json");
                return;
            }
            dump_sd_tensor(out, final_av.first);
            dump_sd_tensor(out, final_av.second);
            out.close();
        }

        size_t file_size = 0;
        try {
            file_size = std::filesystem::file_size(output_path);
        } catch (...) {}

        std::cout << "[ed-dit-server] Sampling successfully completed in " << elapsed_sec
                  << "s, size=" << file_size << " bytes, saved to " << output_path << std::endl;

        json resp = {
            {"status", "success"},
            {"out_path", output_path},
            {"file_size", file_size},
            {"steps", actual_steps},
            {"elapsed_seconds", elapsed_sec},
            {"engine", "pure C++ in-process resident (zero python, zero rust)"}
        };
        res.set_content(resp.dump(), "application/json");
    });

    std::cout << "[ed-dit-server] Server starting on " << host << ":" << port << "..." << std::endl;
    svr.listen(host.c_str(), port);

    return 0;
}

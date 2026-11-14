#include <iostream>
#include <string>
#include <memory>
#include <map>
#include "edge-dit.h"
#include "dit_models/diffusion_model.hpp"
#include "core/runtime/model_loader.h"
#include "core/runtime/model_runtime.hpp"
#include "dit_models/models/minimax_h3_full.hpp"

using namespace edgedit;

int main() {
    ed_context_params_t ctx_params;
    ed_context_params_init(&ctx_params);
    ctx_params.diffusion_model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors";
    
    ModelRuntime runtime;
    std::string err;
    if (!runtime.init(ctx_params, &err)) {
        std::cerr << "[FAILED] runtime.init error: " << err << std::endl;
        return 1;
    }

    ModelLoader loader;
    if (!loader.load_model_files(ctx_params, &err)) {
        std::cerr << "[FAILED] load_model_files error: " << err << std::endl;
        return 1;
    }
    std::cout << "[SUCCESS] load_model_files loaded! Storage count: " << loader.get_tensor_storage_map().size() << std::endl;

    if (!loader.finalize_names_and_version(&err)) {
        std::cerr << "[FAILED] finalize_names_and_version error: " << err << std::endl;
        return 1;
    }
    std::cout << "[SUCCESS] finalize_names_and_version! Version: " << ed_version_name(loader.get_ld_version()) << std::endl;

    std::cout << "Initializing MiniMaxH3Runner..." << std::endl;
    auto runner = std::make_unique<MiniMaxH3::MiniMaxH3Runner>(
        runtime.backend(),
        loader.get_tensor_storage_map(),
        "model.diffusion_model",
        true
    );
    std::cout << "[SUCCESS] MiniMaxH3Runner constructed successfully!" << std::endl;

    size_t graph_budget = 200ULL * 1024 * 1024;
    runner->set_max_graph_vram_bytes(graph_budget);
    runner->alloc_params_buffer();

    std::map<std::string, ggml_tensor*> tensors;
    runner->get_param_tensors(tensors, "model.diffusion_model");
    std::cout << "Runner param tensors count: " << tensors.size() << std::endl;

    std::cout << "Binding weights from safetensors into resident pool..." << std::endl;
    if (!loader.bind_weights(tensors, {}, 8, false, &err)) {
        std::cerr << "[FAILED] bind_weights error: " << err << std::endl;
        return 1;
    }
    std::cout << "[SUCCESS] bind_weights SUCCEEDED! All weights bound into memory!" << std::endl;
    return 0;
}

#include <iostream>
#include <vector>
#include <string>
#include "utils/model_io/safetensors_io.h"
#include "utils/model_io/gguf_io.h"

int main() {
    std::vector<TensorStorage> storages_st;
    std::string err;
    std::string st_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors";
    if (read_safetensors_file(st_path, storages_st, &err)) {
        std::cout << "[SUCCESS] read_safetensors_file read " << storages_st.size() << " tensors!" << std::endl;
    } else {
        std::cout << "[FAILED] read_safetensors_file error: " << err << std::endl;
    }

    std::vector<TensorStorage> storages_gguf;
    std::string gguf_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_w4a8.gguf";
    if (read_gguf_file(gguf_path, storages_gguf, &err)) {
        std::cout << "[SUCCESS] read_gguf_file read " << storages_gguf.size() << " tensors!" << std::endl;
    } else {
        std::cout << "[FAILED] read_gguf_file error: " << err << std::endl;
    }
    return 0;
}

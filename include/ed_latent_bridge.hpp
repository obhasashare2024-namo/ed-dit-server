#pragma once
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <iostream>
#include "utils/tensor.hpp"

namespace edgedit {

struct EdLatentHeader {
    uint32_t magic = 0x48334C54; // "H3LT"
    uint32_t dims = 0;
    int64_t  shape[8] = {0};
    uint64_t num_elements = 0;
};

inline void dump_sd_tensor(std::ofstream& out, const sd::Tensor<float>& t) {
    EdLatentHeader hdr;
    hdr.dims = static_cast<uint32_t>(t.shape().size());
    for (size_t i = 0; i < 8; ++i) {
        hdr.shape[i] = i < t.shape().size() ? t.shape()[i] : 0;
    }
    hdr.num_elements = static_cast<uint64_t>(t.numel());
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(EdLatentHeader));
    out.write(reinterpret_cast<const char*>(t.data()), hdr.num_elements * sizeof(float));
}

inline sd::Tensor<float> load_sd_tensor(std::ifstream& in) {
    EdLatentHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(EdLatentHeader));
    if (!in || hdr.magic != 0x48334C54) {
        throw std::runtime_error("Invalid .edlat file header");
    }
    std::vector<int64_t> shape(hdr.shape, hdr.shape + hdr.dims);
    std::vector<float> data(hdr.num_elements);
    in.read(reinterpret_cast<char*>(data.data()), hdr.num_elements * sizeof(float));
    if (!in) {
        throw std::runtime_error("Truncated .edlat tensor payload");
    }
    return sd::Tensor<float>(std::move(shape), std::move(data));
}

} // namespace edgedit

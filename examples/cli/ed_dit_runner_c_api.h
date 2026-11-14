#ifndef ED_DIT_RUNNER_C_API_H
#define ED_DIT_RUNNER_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* ed_runner_handle_t;

ed_runner_handle_t ed_runner_create(
    const char* model_path,
    int n_threads,
    int budget_mb,
    int enable_flash_attn
);

void ed_runner_set_budget(
    ed_runner_handle_t handle,
    int budget_mb
);

int ed_runner_sample(
    ed_runner_handle_t handle,
    const char* input_path,
    const char* output_path,
    int steps,
    uint64_t seed,
    float* elapsed_sec_out
);

void ed_runner_free(ed_runner_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif // ED_DIT_RUNNER_C_API_H

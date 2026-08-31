#ifndef CODON_PLL_H
#define CODON_PLL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Legacy single-model API (kept for backward compatibility, not used by Rust)
void* codon_model_init(const char* model_path);
void codon_model_free(void* model);
double codon_evaluate_pll(const int64_t* seq, int len, void* model);

// New batched inference API
void* codon_batch_service_init(const char* model_path, int max_batch_size, int timeout_ms);
void codon_batch_service_free(void* service);
double codon_batch_evaluate(void* service, const int64_t* seq, int len);

#ifdef __cplusplus
}
#endif

#endif

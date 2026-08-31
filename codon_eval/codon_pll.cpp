#include "../include/codon_pll.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <iostream>
#include <cmath>
#include <limits>
#include <string>

static const int64_t MAX_SEQ_LEN = 2048;

// ---------------------------------------------------------------------------
// BatchService: GPU batched inference for CodonTransformer PLL
// ---------------------------------------------------------------------------

struct PendingRequest {
    std::vector<int64_t> seq;
    double* result_ptr;
    bool* done_ptr;
    std::condition_variable* cv_ptr;
    std::mutex* mtx_ptr;
};

class BatchService {
public:
    BatchService(const std::string& model_path, int max_batch_size, int timeout_ms)
        : device_(torch::kCPU), max_batch_size_(max_batch_size), timeout_ms_(timeout_ms), running_(true), use_three_inputs_(false) {
        
        // Prefer CUDA, fallback to CPU
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
            std::cerr << "[codon_pll] Using CUDA device" << std::endl;
        } else {
            device_ = torch::Device(torch::kCPU);
            std::cerr << "[codon_pll] CUDA not available, using CPU" << std::endl;
        }
#else
        device_ = torch::Device(torch::kCPU);
        std::cerr << "[codon_pll] Compiled without CUDA support, using CPU" << std::endl;
#endif

        try {
            // Work around a known libtorch/cu118 bug where the JIT fuser emits
            // invalid CUDA literals like `-3.402823466385289e+38.f`, causing
            // nvrtc compilation failure (pytorch/pytorch#107503). Disabling the
            // TensorExpr and NVFuser GPU fusers avoids the bad kernel generation.
            torch::jit::setTensorExprFuserEnabled(false);
            torch::jit::fuser::cuda::setEnabled(false);

            // Load to CPU first, then transfer to target device
            module_ = torch::jit::load(model_path, torch::kCPU);
            module_.to(device_);
            module_.eval();
        } catch (const c10::Error& e) {
            std::cerr << "[codon_pll] Failed to load model: " << e.what() << std::endl;
            throw;
        }

        // Detect model forward signature: some models expect (input_ids) only,
        // others expect (input_ids, attention_mask, token_type_ids).
        try {
            auto method = module_.get_method("forward");
            auto schema = method.function().getSchema();
            size_t num_args = schema.arguments().size(); // includes 'self'
            if (num_args >= 4) {
                use_three_inputs_ = true;
                std::cerr << "[codon_pll] Model forward accepts 3 input tensors" << std::endl;
            } else {
                use_three_inputs_ = false;
                std::cerr << "[codon_pll] Model forward accepts 1 input tensor" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[codon_pll] Could not inspect forward signature, defaulting to 1 input tensor" << std::endl;
            use_three_inputs_ = false;
        }

        inference_thread_ = std::thread(&BatchService::inference_loop, this);
    }

    ~BatchService() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            running_ = false;
        }
        cv_producer_.notify_all();
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
    }

    double evaluate(const int64_t* seq, int len) {
        if (!seq || len < 3) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (len > MAX_SEQ_LEN) {
            std::cerr << "[codon_pll] Sequence length " << len 
                      << " exceeds max " << MAX_SEQ_LEN << std::endl;
            return std::numeric_limits<double>::quiet_NaN();
        }

        double result = std::numeric_limits<double>::quiet_NaN();
        std::mutex local_mtx;
        std::condition_variable local_cv;
        bool done = false;

        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push(PendingRequest{
                std::vector<int64_t>(seq, seq + len),
                &result,
                &done,
                &local_cv,
                &local_mtx
            });
        }
        cv_producer_.notify_one();

        // Block until the inference thread fills in the result
        std::unique_lock<std::mutex> lock(local_mtx);
        local_cv.wait(lock, [&] { return done; });
        return result;
    }

private:
    void inference_loop() {
        while (true) {
            std::vector<PendingRequest> batch;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_producer_.wait_for(lock, std::chrono::milliseconds(timeout_ms_),
                    [&] { return !queue_.empty() || !running_; });

                if (!running_ && queue_.empty()) {
                    break;
                }

                while (!queue_.empty() && static_cast<int>(batch.size()) < max_batch_size_) {
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop();
                }
            }

            if (batch.empty()) {
                continue;
            }

            std::vector<double> results = batch_inference(batch);
            for (size_t i = 0; i < batch.size(); ++i) {
                *batch[i].result_ptr = results[i];
                {
                    std::lock_guard<std::mutex> lock(*batch[i].mtx_ptr);
                    *batch[i].done_ptr = true;
                }
                batch[i].cv_ptr->notify_one();
            }
        }
    }

    std::vector<double> batch_inference(const std::vector<PendingRequest>& batch) {
        const size_t batch_size = batch.size();
        if (batch_size == 0) {
            return {};
        }

        // Find max sequence length in this batch
        size_t max_len = 0;
        for (const auto& req : batch) {
            if (req.seq.size() > max_len) {
                max_len = req.seq.size();
            }
        }

        // Build padded batch tensor
        std::vector<std::vector<int64_t>> padded_seqs;
        padded_seqs.reserve(batch_size);
        for (const auto& req : batch) {
            std::vector<int64_t> padded = req.seq;
            padded.resize(max_len, 0);  // pad with 0
            padded_seqs.push_back(std::move(padded));
        }

        try {
            torch::NoGradGuard no_grad;

            // Flatten padded sequences and build batch tensor
            std::vector<int64_t> flat;
            std::vector<int64_t> flat_mask;
            flat.reserve(batch_size * max_len);
            flat_mask.reserve(batch_size * max_len);
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < max_len; ++i) {
                    flat.push_back(padded_seqs[b][i]);
                    flat_mask.push_back(i < batch[b].seq.size() ? 1 : 0);
                }
            }
            torch::Tensor input_ids = torch::from_blob(
                flat.data(),
                {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_len)},
                torch::dtype(torch::kLong)
            ).clone().to(device_);
            // input_ids shape: [batch_size, max_len]

            torch::Tensor attention_mask = torch::from_blob(
                flat_mask.data(),
                {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_len)},
                torch::dtype(torch::kLong)
            ).clone().to(device_);

            // token_type_ids: all tokens use organism ID 59 (Homo sapiens).
            // The CodonTransformer model uses token_type_ids to encode the
            // target organism; 59 corresponds to human, which is the default
            // species for the pre-trained human checkpoint.
            torch::Tensor token_type_ids = torch::full(
                {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_len)},
                59,
                torch::dtype(torch::kLong)
            ).to(device_);

            torch::Tensor logits;
            if (use_three_inputs_) {
                logits = module_.forward({input_ids, attention_mask, token_type_ids}).toTensor();
            } else {
                logits = module_.forward({input_ids}).toTensor();
            }
            // logits shape: [batch_size, max_len, vocab_size]

            std::vector<double> results;
            results.reserve(batch_size);

            for (size_t b = 0; b < batch_size; ++b) {
                int64_t seq_len = static_cast<int64_t>(batch[b].seq.size());
                // ── Pseudo Log-Likelihood (PLL) for Masked Language Models ──
                //
                // Standard MLM-PLL formula (Salazar et al., 2020):
                //   PLL(x) = Σ_{t=1}^{T} log P(x_t | x_{\t})
                //
                // where each position t is predicted from the full bidirectional
                // context (all other tokens).  This is NOT the autoregressive
                // (causal) log-likelihood; it assumes the model is BERT-style.
                //
                // We trim [CLS] (position 0) and [SEP] (position seq_len-1),
                // then compute:
                //   log_softmax(logits[1..seq_len-1]) → gather(true_ids[1..seq_len-1]) → sum
                //
                // The logits at each position i predict the token at the same
                // position i (bidirectional), which is correct for MLM-PLL.

                // Extract logits and true ids for this sequence, trimming [CLS] and [SEP]
                // logits[b] shape: [max_len, vocab_size]
                torch::Tensor logits_b = logits[b];
                torch::Tensor logits_trim = logits_b.slice(0, 1, seq_len - 1);
                // true_ids shape: [seq_len]
                torch::Tensor true_ids = input_ids[b].slice(0, 1, seq_len - 1);

                torch::Tensor log_probs = torch::log_softmax(logits_trim, -1);
                torch::Tensor gathered = torch::gather(log_probs, 1, true_ids.unsqueeze(-1));
                torch::Tensor pll = gathered.squeeze(-1).sum(-1);
                results.push_back(pll.item<double>());
            }

            return results;
        } catch (const c10::Error& e) {
            std::cerr << "[codon_pll] Batch inference failed: " << e.what() << std::endl;
            return std::vector<double>(batch_size, std::numeric_limits<double>::quiet_NaN());
        }
    }

    torch::jit::script::Module module_;
    torch::Device device_;
    int max_batch_size_;
    int timeout_ms_;
    bool use_three_inputs_;

    std::thread inference_thread_;
    std::mutex mtx_;
    std::condition_variable cv_producer_;
    std::queue<PendingRequest> queue_;
    bool running_;
};

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

void* codon_batch_service_init(const char* model_path, int max_batch_size, int timeout_ms) {
    try {
        auto* service = new BatchService(model_path, max_batch_size, timeout_ms);
        return static_cast<void*>(service);
    } catch (const c10::Error& e) {
        std::cerr << "[codon_batch_service_init] failed: " << e.what() << std::endl;
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "[codon_batch_service_init] failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void codon_batch_service_free(void* service) {
    if (service) {
        delete static_cast<BatchService*>(service);
    }
}

double codon_batch_evaluate(void* service, const int64_t* seq, int len) {
    if (!service) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<BatchService*>(service)->evaluate(seq, len);
}

/**
 * On-Device Gradient Engine Implementation — V5 Architecture
 *
 * Implements backpropagation and SGD with momentum for the V5
 * 3-layer classification heads (FC1[frozen] -> LN -> GELU -> FC2 ->
 * GELU -> FC3) on ESP32-S3.
 *
 * All memory is pre-allocated in begin() — no dynamic allocations
 * during training to guarantee deterministic, real-time behavior.
 */

#include "gradient_engine.h"
#include "transport/console.h"
#include "../inference/model_weights.h"

// GELU constants
static constexpr float SQRT_2_OVER_PI = 0.7978845608f;  // sqrt(2/pi)
static constexpr float GELU_COEFF     = 0.044715f;
static constexpr float LN_EPS         = 1e-5f;

// Global instance
GradientEngine gradientEngine;

// ============================================================
// GELU helpers (used on stack buffers, not class methods)
// ============================================================

static inline float gelu_forward(float x) {
    float x3 = x * x * x;
    float a = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    return 0.5f * x * (1.0f + tanhf(a));
}

static inline float gelu_derivative(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float a = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    float tanh_a = tanhf(a);
    float a_prime = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF * x2);
    return 0.5f * (1.0f + tanh_a) + 0.5f * x * (1.0f - tanh_a * tanh_a) * a_prime;
}

// ============================================================
// Constructor / Destructor
// ============================================================

GradientEngine::GradientEngine()
    : _initialized(false),
      _adapted(false),
      _total_trainable_params(0),
      _generation(0) {
    for (int i = 0; i < 3; i++) {
        _heads[i] = {};
    }
}

GradientEngine::~GradientEngine() {
    for (int i = 0; i < 3; i++) {
        freeTaskHead(_heads[i]);
    }
}

// ============================================================
// Initialization
// ============================================================

bool GradientEngine::begin() {
    if (_initialized) return true;

    CONSOLE.println("[GradientEngine] Initializing V5 on-device training engine...");

    // Allocate Activity head: [64] -> FC1[64] -> LN[64] -> GELU -> FC2[32] -> GELU -> FC3[4]
    if (!allocTaskHead(_heads[0], 64, 64, 32, 4,
                       activity_head_fc1_weight, activity_head_fc1_bias,
                       activity_head_ln_weight,  activity_head_ln_bias,
                       activity_head_fc2_weight, activity_head_fc2_bias,
                       activity_head_fc3_weight, activity_head_fc3_bias)) {
        CONSOLE.println("[GradientEngine] ERROR: Failed to allocate activity head");
        return false;
    }

    // Allocate Stress head: [64] -> FC1[48] -> LN[48] -> GELU -> FC2[24] -> GELU -> FC3[2]
    if (!allocTaskHead(_heads[1], 64, 48, 24, 2,
                       stress_head_fc1_weight, stress_head_fc1_bias,
                       stress_head_ln_weight,  stress_head_ln_bias,
                       stress_head_fc2_weight, stress_head_fc2_bias,
                       stress_head_fc3_weight, stress_head_fc3_bias)) {
        CONSOLE.println("[GradientEngine] ERROR: Failed to allocate stress head");
        return false;
    }

    // Allocate Arrhythmia head: [64] -> FC1[48] -> LN[48] -> GELU -> FC2[24] -> GELU -> FC3[2]
    if (!allocTaskHead(_heads[2], 64, 48, 24, 2,
                       arrhythmia_head_fc1_weight, arrhythmia_head_fc1_bias,
                       arrhythmia_head_ln_weight,  arrhythmia_head_ln_bias,
                       arrhythmia_head_fc2_weight, arrhythmia_head_fc2_bias,
                       arrhythmia_head_fc3_weight, arrhythmia_head_fc3_bias)) {
        CONSOLE.println("[GradientEngine] ERROR: Failed to allocate arrhythmia head");
        return false;
    }

    // Count trainable parameters: LN + FC2 + FC3 for each head (FC1 is frozen)
    _total_trainable_params = 0;
    for (int i = 0; i < 3; i++) {
        const auto& h = _heads[i];
        // LN: weight[hidden1] + bias[hidden1]
        _total_trainable_params += h.hidden1_dim + h.hidden1_dim;
        // FC2: weight[hidden2 * hidden1] + bias[hidden2]
        _total_trainable_params += h.hidden1_dim * h.hidden2_dim + h.hidden2_dim;
        // FC3: weight[output * hidden2] + bias[output]
        _total_trainable_params += h.hidden2_dim * h.output_dim + h.output_dim;
    }

    size_t total_bytes = 0;
    for (int i = 0; i < 3; i++) {
        total_bytes += _heads[i].totalBytes();
    }

    CONSOLE.printf("[GradientEngine] Allocated %.1f KB for training buffers\n",
                  total_bytes / 1024.0f);
    CONSOLE.printf("[GradientEngine] Trainable parameters: %lu (FC1 frozen)\n",
                  (unsigned long)_total_trainable_params);

    _initialized = true;
    CONSOLE.println("[GradientEngine] V5 on-device training engine ready.");
    return true;
}

// ============================================================
// Task Head Allocation
// ============================================================

bool GradientEngine::allocTaskHead(
    TaskHeadWeights& head,
    int input_dim, int hidden1_dim, int hidden2_dim, int output_dim,
    const float* flash_fc1_w, const float* flash_fc1_b,
    const float* flash_ln_w,  const float* flash_ln_b,
    const float* flash_fc2_w, const float* flash_fc2_b,
    const float* flash_fc3_w, const float* flash_fc3_b)
{
    head.input_dim   = input_dim;
    head.hidden1_dim = hidden1_dim;
    head.hidden2_dim = hidden2_dim;
    head.output_dim  = output_dim;

    const size_t fc1_w_size = input_dim * hidden1_dim;
    const size_t fc2_w_size = hidden1_dim * hidden2_dim;
    const size_t fc3_w_size = hidden2_dim * output_dim;

    // Helper macro: allocate in PSRAM, zero-initialize
    #define ALLOC_PSRAM(ptr, count) do { \
        ptr = (float*)ps_malloc((count) * sizeof(float)); \
        if (!ptr) return false; \
        memset(ptr, 0, (count) * sizeof(float)); \
    } while(0)

    // FC1 weights (frozen copy from flash)
    ALLOC_PSRAM(head.fc1_weight, fc1_w_size);
    ALLOC_PSRAM(head.fc1_bias, hidden1_dim);
    memcpy(head.fc1_weight, flash_fc1_w, fc1_w_size * sizeof(float));
    memcpy(head.fc1_bias, flash_fc1_b, hidden1_dim * sizeof(float));

    // LN weights (trainable, copied from flash)
    ALLOC_PSRAM(head.ln_weight, hidden1_dim);
    ALLOC_PSRAM(head.ln_bias, hidden1_dim);
    memcpy(head.ln_weight, flash_ln_w, hidden1_dim * sizeof(float));
    memcpy(head.ln_bias, flash_ln_b, hidden1_dim * sizeof(float));

    // FC2 weights (trainable, copied from flash)
    ALLOC_PSRAM(head.fc2_weight, fc2_w_size);
    ALLOC_PSRAM(head.fc2_bias, hidden2_dim);
    memcpy(head.fc2_weight, flash_fc2_w, fc2_w_size * sizeof(float));
    memcpy(head.fc2_bias, flash_fc2_b, hidden2_dim * sizeof(float));

    // FC3 weights (trainable, copied from flash)
    ALLOC_PSRAM(head.fc3_weight, fc3_w_size);
    ALLOC_PSRAM(head.fc3_bias, output_dim);
    memcpy(head.fc3_weight, flash_fc3_w, fc3_w_size * sizeof(float));
    memcpy(head.fc3_bias, flash_fc3_b, output_dim * sizeof(float));

    // Momentum buffers for trainable layers only (LN + FC2 + FC3)
    ALLOC_PSRAM(head.ln_weight_vel, hidden1_dim);
    ALLOC_PSRAM(head.ln_bias_vel, hidden1_dim);
    ALLOC_PSRAM(head.fc2_weight_vel, fc2_w_size);
    ALLOC_PSRAM(head.fc2_bias_vel, hidden2_dim);
    ALLOC_PSRAM(head.fc3_weight_vel, fc3_w_size);
    ALLOC_PSRAM(head.fc3_bias_vel, output_dim);

    // Gradient buffers for trainable layers only (LN + FC2 + FC3)
    ALLOC_PSRAM(head.ln_weight_grad, hidden1_dim);
    ALLOC_PSRAM(head.ln_bias_grad, hidden1_dim);
    ALLOC_PSRAM(head.fc2_weight_grad, fc2_w_size);
    ALLOC_PSRAM(head.fc2_bias_grad, hidden2_dim);
    ALLOC_PSRAM(head.fc3_weight_grad, fc3_w_size);
    ALLOC_PSRAM(head.fc3_bias_grad, output_dim);

    // Activation caches for backprop
    ALLOC_PSRAM(head.fc1_output, hidden1_dim);     // pre-LN
    ALLOC_PSRAM(head.ln_output, hidden1_dim);       // x_hat (normalized)
    ALLOC_PSRAM(head.fc1_activated, hidden1_dim);   // post-GELU (after LN+GELU)
    ALLOC_PSRAM(head.fc2_output, hidden2_dim);      // pre-GELU
    ALLOC_PSRAM(head.fc2_activated, hidden2_dim);   // post-GELU
    ALLOC_PSRAM(head.fc3_output, output_dim);       // logits

    head.ln_mean = 0.0f;
    head.ln_inv_std = 0.0f;

    #undef ALLOC_PSRAM
    return true;
}

void GradientEngine::freeTaskHead(TaskHeadWeights& head) {
    // Weights
    free(head.fc1_weight);  free(head.fc1_bias);
    free(head.ln_weight);   free(head.ln_bias);
    free(head.fc2_weight);  free(head.fc2_bias);
    free(head.fc3_weight);  free(head.fc3_bias);
    // Momentum
    free(head.ln_weight_vel);  free(head.ln_bias_vel);
    free(head.fc2_weight_vel); free(head.fc2_bias_vel);
    free(head.fc3_weight_vel); free(head.fc3_bias_vel);
    // Gradients
    free(head.ln_weight_grad);  free(head.ln_bias_grad);
    free(head.fc2_weight_grad); free(head.fc2_bias_grad);
    free(head.fc3_weight_grad); free(head.fc3_bias_grad);
    // Activations
    free(head.fc1_output);   free(head.ln_output);
    free(head.fc1_activated);
    free(head.fc2_output);   free(head.fc2_activated);
    free(head.fc3_output);
    head = {};
}

// ============================================================
// Forward Pass (with activation caching for backprop)
// ============================================================
// FC1 -> LayerNorm -> GELU -> FC2 -> GELU -> FC3

void GradientEngine::headForward(TaskHeadWeights& head, const float* input) {
    // --- Step 1: FC1 (frozen) --- [input_dim] -> [hidden1_dim]
    for (int o = 0; o < head.hidden1_dim; o++) {
        float sum = head.fc1_bias[o];
        for (int i = 0; i < head.input_dim; i++) {
            sum += input[i] * head.fc1_weight[o * head.input_dim + i];
        }
        head.fc1_output[o] = sum;  // Cache pre-LN output
    }

    // --- Step 2: LayerNorm --- [hidden1_dim] -> [hidden1_dim]
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < head.hidden1_dim; i++) {
        mean += head.fc1_output[i];
    }
    mean /= head.hidden1_dim;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < head.hidden1_dim; i++) {
        float diff = head.fc1_output[i] - mean;
        var += diff * diff;
    }
    var /= head.hidden1_dim;

    float inv_std = 1.0f / sqrtf(var + LN_EPS);

    // Cache for backprop
    head.ln_mean = mean;
    head.ln_inv_std = inv_std;

    // Normalize, scale, shift
    for (int i = 0; i < head.hidden1_dim; i++) {
        float x_hat = (head.fc1_output[i] - mean) * inv_std;
        head.ln_output[i] = x_hat;  // Cache normalized (before scale+shift)
        float ln_out = head.ln_weight[i] * x_hat + head.ln_bias[i];
        // GELU activation
        head.fc1_activated[i] = gelu_forward(ln_out);
    }

    // --- Step 3: FC2 --- [hidden1_dim] -> [hidden2_dim]
    for (int o = 0; o < head.hidden2_dim; o++) {
        float sum = head.fc2_bias[o];
        for (int i = 0; i < head.hidden1_dim; i++) {
            sum += head.fc1_activated[i] * head.fc2_weight[o * head.hidden1_dim + i];
        }
        head.fc2_output[o] = sum;  // Cache pre-GELU
        head.fc2_activated[o] = gelu_forward(sum);  // GELU activation
    }

    // --- Step 4: FC3 --- [hidden2_dim] -> [output_dim]
    for (int o = 0; o < head.output_dim; o++) {
        float sum = head.fc3_bias[o];
        for (int i = 0; i < head.hidden2_dim; i++) {
            sum += head.fc2_activated[i] * head.fc3_weight[o * head.hidden2_dim + i];
        }
        head.fc3_output[o] = sum;  // Cache logits (pre-softmax)
    }
}

// ============================================================
// Backward Pass: Cross-Entropy Loss Gradient
// ============================================================
// Backprop: CE+Softmax -> FC3 -> GELU -> FC2 -> GELU -> LN -> STOP (FC1 frozen)

float GradientEngine::headBackward(TaskHeadWeights& head,
                                    const float* input,
                                    uint8_t true_label) {
    // --- Softmax on logits ---
    float probs[16];  // Max output_dim across all heads is 4
    float max_logit = head.fc3_output[0];
    for (int i = 1; i < head.output_dim; i++) {
        if (head.fc3_output[i] > max_logit) max_logit = head.fc3_output[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < head.output_dim; i++) {
        probs[i] = expf(head.fc3_output[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < head.output_dim; i++) {
        probs[i] /= sum_exp;
    }

    // Cross-entropy loss: -log(probs[true_label])
    float loss = -logf(fmaxf(probs[true_label], 1e-7f));

    // --- dL/d(logits) = probs - one_hot(true_label) ---
    float d_logits[16];
    for (int i = 0; i < head.output_dim; i++) {
        d_logits[i] = probs[i];
    }
    d_logits[true_label] -= 1.0f;

    // --- FC3 gradients ---
    // dL/dW3[o][i] = d_logits[o] * fc2_activated[i]
    // dL/db3[o]    = d_logits[o]
    for (int o = 0; o < head.output_dim; o++) {
        head.fc3_bias_grad[o] = d_logits[o];
        for (int i = 0; i < head.hidden2_dim; i++) {
            head.fc3_weight_grad[o * head.hidden2_dim + i] =
                d_logits[o] * head.fc2_activated[i];
        }
    }

    // --- Backprop through FC3 to d_fc2_activated ---
    // d_fc2_activated[i] = sum_o(d_logits[o] * W3[o][i])
    float d_fc2_act[64];  // max hidden2_dim = 32
    for (int i = 0; i < head.hidden2_dim; i++) {
        float sum = 0.0f;
        for (int o = 0; o < head.output_dim; o++) {
            sum += d_logits[o] * head.fc3_weight[o * head.hidden2_dim + i];
        }
        d_fc2_act[i] = sum;
    }

    // --- Backprop through GELU (FC2 output) ---
    // d_fc2_output[i] = d_fc2_act[i] * gelu'(fc2_output[i])
    float d_fc2_out[64];  // max hidden2_dim = 32
    for (int i = 0; i < head.hidden2_dim; i++) {
        d_fc2_out[i] = d_fc2_act[i] * gelu_derivative(head.fc2_output[i]);
    }

    // --- FC2 gradients ---
    // dL/dW2[o][i] = d_fc2_out[o] * fc1_activated[i]
    // dL/db2[o]    = d_fc2_out[o]
    for (int o = 0; o < head.hidden2_dim; o++) {
        head.fc2_bias_grad[o] = d_fc2_out[o];
        for (int i = 0; i < head.hidden1_dim; i++) {
            head.fc2_weight_grad[o * head.hidden1_dim + i] =
                d_fc2_out[o] * head.fc1_activated[i];
        }
    }

    // --- Backprop through FC2 to d_fc1_activated ---
    // d_fc1_activated[i] = sum_o(d_fc2_out[o] * W2[o][i])
    float d_fc1_act[64];  // max hidden1_dim = 64
    for (int i = 0; i < head.hidden1_dim; i++) {
        float sum = 0.0f;
        for (int o = 0; o < head.hidden2_dim; o++) {
            sum += d_fc2_out[o] * head.fc2_weight[o * head.hidden1_dim + i];
        }
        d_fc1_act[i] = sum;
    }

    // --- Backprop through GELU (after LN) ---
    // fc1_activated = GELU(ln_weight * x_hat + ln_bias)
    // We need the pre-GELU values: ln_weight[i] * ln_output[i] + ln_bias[i]
    float d_ln_out[64];  // max hidden1_dim = 64
    for (int i = 0; i < head.hidden1_dim; i++) {
        float pre_gelu = head.ln_weight[i] * head.ln_output[i] + head.ln_bias[i];
        d_ln_out[i] = d_fc1_act[i] * gelu_derivative(pre_gelu);
    }

    // --- LN gradients ---
    // dL/d(ln_weight[i]) = d_ln_out[i] * x_hat[i]
    // dL/d(ln_bias[i])   = d_ln_out[i]
    for (int i = 0; i < head.hidden1_dim; i++) {
        head.ln_weight_grad[i] = d_ln_out[i] * head.ln_output[i];
        head.ln_bias_grad[i]   = d_ln_out[i];
    }

    // STOP — FC1 is frozen, no further backprop needed

    return loss;
}

// ============================================================
// Gradient Clipping (L2 Norm)
// ============================================================

float GradientEngine::clipGradients(TaskHeadWeights& head) {
    // Compute global gradient L2 norm across all trainable parameters
    float norm_sq = 0.0f;

    // LN gradients
    for (int i = 0; i < head.hidden1_dim; i++) {
        norm_sq += head.ln_weight_grad[i] * head.ln_weight_grad[i];
        norm_sq += head.ln_bias_grad[i]   * head.ln_bias_grad[i];
    }

    // FC2 gradients
    for (int i = 0; i < head.hidden1_dim * head.hidden2_dim; i++) {
        norm_sq += head.fc2_weight_grad[i] * head.fc2_weight_grad[i];
    }
    for (int i = 0; i < head.hidden2_dim; i++) {
        norm_sq += head.fc2_bias_grad[i] * head.fc2_bias_grad[i];
    }

    // FC3 gradients
    for (int i = 0; i < head.hidden2_dim * head.output_dim; i++) {
        norm_sq += head.fc3_weight_grad[i] * head.fc3_weight_grad[i];
    }
    for (int i = 0; i < head.output_dim; i++) {
        norm_sq += head.fc3_bias_grad[i] * head.fc3_bias_grad[i];
    }

    float norm = sqrtf(norm_sq + 1e-8f);

    // Clip if exceeds threshold
    if (norm > AdaptiveConfig::GRADIENT_CLIP_NORM) {
        float scale = AdaptiveConfig::GRADIENT_CLIP_NORM / norm;

        for (int i = 0; i < head.hidden1_dim; i++) {
            head.ln_weight_grad[i] *= scale;
            head.ln_bias_grad[i]   *= scale;
        }
        for (int i = 0; i < head.hidden1_dim * head.hidden2_dim; i++) {
            head.fc2_weight_grad[i] *= scale;
        }
        for (int i = 0; i < head.hidden2_dim; i++) {
            head.fc2_bias_grad[i] *= scale;
        }
        for (int i = 0; i < head.hidden2_dim * head.output_dim; i++) {
            head.fc3_weight_grad[i] *= scale;
        }
        for (int i = 0; i < head.output_dim; i++) {
            head.fc3_bias_grad[i] *= scale;
        }
    }

    return norm;
}

// ============================================================
// SGD with Momentum + Weight Clamping
// ============================================================

void GradientEngine::applyGradients(TaskHeadWeights& head, float lr) {
    const float mu = AdaptiveConfig::MOMENTUM;

    // Helper: SGD with momentum for a parameter tensor
    // v = mu * v + grad
    // w = w - lr * v
    auto sgd_update = [lr, mu](float* weight, float* velocity,
                                const float* grad, int size) {
        for (int i = 0; i < size; i++) {
            velocity[i] = mu * velocity[i] + grad[i];
            weight[i] -= lr * velocity[i];
        }
    };

    // LN (trainable)
    sgd_update(head.ln_weight, head.ln_weight_vel,
               head.ln_weight_grad, head.hidden1_dim);
    sgd_update(head.ln_bias, head.ln_bias_vel,
               head.ln_bias_grad, head.hidden1_dim);

    // FC2 (trainable)
    sgd_update(head.fc2_weight, head.fc2_weight_vel,
               head.fc2_weight_grad, head.hidden1_dim * head.hidden2_dim);
    sgd_update(head.fc2_bias, head.fc2_bias_vel,
               head.fc2_bias_grad, head.hidden2_dim);

    // FC3 (trainable)
    sgd_update(head.fc3_weight, head.fc3_weight_vel,
               head.fc3_weight_grad, head.hidden2_dim * head.output_dim);
    sgd_update(head.fc3_bias, head.fc3_bias_vel,
               head.fc3_bias_grad, head.output_dim);

    // NO FC1 updates — frozen
}

void GradientEngine::clampWeights(TaskHeadWeights& head) {
    const float max_abs = AdaptiveConfig::WEIGHT_MAX_ABS;

    auto clamp = [max_abs](float* data, int size) {
        for (int i = 0; i < size; i++) {
            if (data[i] > max_abs) data[i] = max_abs;
            else if (data[i] < -max_abs) data[i] = -max_abs;
        }
    };

    // LN
    clamp(head.ln_weight, head.hidden1_dim);
    clamp(head.ln_bias, head.hidden1_dim);

    // FC2
    clamp(head.fc2_weight, head.hidden1_dim * head.hidden2_dim);
    clamp(head.fc2_bias, head.hidden2_dim);

    // FC3
    clamp(head.fc3_weight, head.hidden2_dim * head.output_dim);
    clamp(head.fc3_bias, head.output_dim);

    // FC1 not clamped — frozen
}

// ============================================================
// NaN/Inf Check
// ============================================================

bool GradientEngine::hasNanInf(const float* data, int size) const {
    for (int i = 0; i < size; i++) {
        if (isnan(data[i]) || isinf(data[i])) return true;
    }
    return false;
}

// ============================================================
// Training Step
// ============================================================

TrainingResult GradientEngine::trainStep(
    uint8_t task_id,
    const float* features,
    uint8_t true_label,
    float learning_rate)
{
    TrainingResult result = {};
    result.success = false;

    if (!_initialized || task_id > 2) return result;

    TaskHeadWeights& head = _heads[task_id];

    // Validate label
    if (true_label >= head.output_dim) {
        CONSOLE.printf("[GradientEngine] Invalid label %d for task %d (max %d)\n",
                      true_label, task_id, head.output_dim - 1);
        return result;
    }

    unsigned long t0 = micros();

    // 1. Forward pass (caches activations)
    headForward(head, features);

    // 2. Check for NaN in forward pass (check logits = fc3_output)
    if (hasNanInf(head.fc3_output, head.output_dim)) {
        CONSOLE.println("[GradientEngine] NaN/Inf in forward pass — aborting step");
        return result;
    }

    // 3. Backward pass (computes gradients, returns loss)
    float loss = headBackward(head, features, true_label);

    // 4. Clip gradients
    float grad_norm = clipGradients(head);

    // 5. Check for NaN in gradients (all trainable layers)
    if (hasNanInf(head.ln_weight_grad, head.hidden1_dim) ||
        hasNanInf(head.ln_bias_grad, head.hidden1_dim) ||
        hasNanInf(head.fc2_weight_grad, head.hidden1_dim * head.hidden2_dim) ||
        hasNanInf(head.fc2_bias_grad, head.hidden2_dim) ||
        hasNanInf(head.fc3_weight_grad, head.hidden2_dim * head.output_dim) ||
        hasNanInf(head.fc3_bias_grad, head.output_dim)) {
        CONSOLE.println("[GradientEngine] NaN/Inf in gradients — aborting step");
        return result;
    }

    // 6. Apply SGD with momentum
    applyGradients(head, learning_rate);

    // 7. Clamp weight magnitudes
    clampWeights(head);

    // 8. Final NaN check on updated weights
    if (hasNanInf(head.ln_weight, head.hidden1_dim) ||
        hasNanInf(head.fc2_weight, head.hidden1_dim * head.hidden2_dim) ||
        hasNanInf(head.fc3_weight, head.hidden2_dim * head.output_dim)) {
        CONSOLE.println("[GradientEngine] NaN/Inf after update — rolling back");
        resetToOriginal(task_id);
        return result;
    }

    // Count updated parameters (LN + FC2 + FC3)
    result.params_updated = head.hidden1_dim + head.hidden1_dim;  // LN weight + bias
    result.params_updated += head.hidden1_dim * head.hidden2_dim + head.hidden2_dim;  // FC2
    result.params_updated += head.hidden2_dim * head.output_dim + head.output_dim;    // FC3

    result.loss = loss;
    result.gradient_norm = grad_norm;
    result.duration_ms = (micros() - t0) / 1000.0f;
    result.success = true;

    // Mark that we have adapted weights (inference should use them)
    _adapted = true;

    return result;
}

// ============================================================
// Inference with Adapted Weights
// ============================================================
// Full 3-layer forward: FC1 -> LN -> GELU -> FC2 -> GELU -> FC3 -> Softmax

void GradientEngine::inferWithAdapted(
    uint8_t task_id,
    const float* features,
    float* output)
{
    if (!_initialized || task_id > 2) return;

    TaskHeadWeights& head = _heads[task_id];

    // --- FC1 (frozen weights) ---
    float fc1_out[64];  // max hidden1_dim = 64
    for (int o = 0; o < head.hidden1_dim; o++) {
        float sum = head.fc1_bias[o];
        for (int i = 0; i < head.input_dim; i++) {
            sum += features[i] * head.fc1_weight[o * head.input_dim + i];
        }
        fc1_out[o] = sum;
    }

    // --- LayerNorm ---
    float mean = 0.0f;
    for (int i = 0; i < head.hidden1_dim; i++) {
        mean += fc1_out[i];
    }
    mean /= head.hidden1_dim;

    float var = 0.0f;
    for (int i = 0; i < head.hidden1_dim; i++) {
        float diff = fc1_out[i] - mean;
        var += diff * diff;
    }
    var /= head.hidden1_dim;

    float inv_std = 1.0f / sqrtf(var + LN_EPS);

    // LN + GELU
    float activated1[64];  // max hidden1_dim = 64
    for (int i = 0; i < head.hidden1_dim; i++) {
        float x_hat = (fc1_out[i] - mean) * inv_std;
        float ln_out = head.ln_weight[i] * x_hat + head.ln_bias[i];
        activated1[i] = gelu_forward(ln_out);
    }

    // --- FC2 + GELU ---
    float activated2[64];  // max hidden2_dim = 32
    for (int o = 0; o < head.hidden2_dim; o++) {
        float sum = head.fc2_bias[o];
        for (int i = 0; i < head.hidden1_dim; i++) {
            sum += activated1[i] * head.fc2_weight[o * head.hidden1_dim + i];
        }
        activated2[o] = gelu_forward(sum);
    }

    // --- FC3 (logits) ---
    for (int o = 0; o < head.output_dim; o++) {
        float sum = head.fc3_bias[o];
        for (int i = 0; i < head.hidden2_dim; i++) {
            sum += activated2[i] * head.fc3_weight[o * head.hidden2_dim + i];
        }
        output[o] = sum;
    }

    // --- Softmax ---
    float max_val = output[0];
    for (int i = 1; i < head.output_dim; i++) {
        if (output[i] > max_val) max_val = output[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < head.output_dim; i++) {
        output[i] = expf(output[i] - max_val);
        sum_exp += output[i];
    }
    for (int i = 0; i < head.output_dim; i++) {
        output[i] /= sum_exp;
    }
}

// ============================================================
// Weight Snapshot / Restore (for Model Store)
// ============================================================
// Serialized format: fc1_w | fc1_b | ln_w | ln_b | fc2_w | fc2_b | fc3_w | fc3_b

bool GradientEngine::snapshotWeights(uint8_t task_id, float* dest,
                                      size_t max_bytes) const {
    if (!_initialized || task_id > 2) return false;
    const TaskHeadWeights& h = _heads[task_id];

    size_t needed = (h.input_dim * h.hidden1_dim + h.hidden1_dim +   // FC1
                     h.hidden1_dim + h.hidden1_dim +                  // LN
                     h.hidden1_dim * h.hidden2_dim + h.hidden2_dim +  // FC2
                     h.hidden2_dim * h.output_dim + h.output_dim      // FC3
                    ) * sizeof(float);
    if (max_bytes < needed) return false;

    float* ptr = dest;

    // FC1 (frozen, but included in snapshot for completeness)
    memcpy(ptr, h.fc1_weight, h.input_dim * h.hidden1_dim * sizeof(float));
    ptr += h.input_dim * h.hidden1_dim;
    memcpy(ptr, h.fc1_bias, h.hidden1_dim * sizeof(float));
    ptr += h.hidden1_dim;

    // LN
    memcpy(ptr, h.ln_weight, h.hidden1_dim * sizeof(float));
    ptr += h.hidden1_dim;
    memcpy(ptr, h.ln_bias, h.hidden1_dim * sizeof(float));
    ptr += h.hidden1_dim;

    // FC2
    memcpy(ptr, h.fc2_weight, h.hidden1_dim * h.hidden2_dim * sizeof(float));
    ptr += h.hidden1_dim * h.hidden2_dim;
    memcpy(ptr, h.fc2_bias, h.hidden2_dim * sizeof(float));
    ptr += h.hidden2_dim;

    // FC3
    memcpy(ptr, h.fc3_weight, h.hidden2_dim * h.output_dim * sizeof(float));
    ptr += h.hidden2_dim * h.output_dim;
    memcpy(ptr, h.fc3_bias, h.output_dim * sizeof(float));

    return true;
}

bool GradientEngine::restoreWeights(uint8_t task_id, const float* src,
                                     size_t num_bytes) {
    if (!_initialized || task_id > 2) return false;
    TaskHeadWeights& h = _heads[task_id];

    size_t expected = (h.input_dim * h.hidden1_dim + h.hidden1_dim +
                       h.hidden1_dim + h.hidden1_dim +
                       h.hidden1_dim * h.hidden2_dim + h.hidden2_dim +
                       h.hidden2_dim * h.output_dim + h.output_dim
                      ) * sizeof(float);
    if (num_bytes != expected) return false;

    const float* ptr = src;

    // FC1
    memcpy(h.fc1_weight, ptr, h.input_dim * h.hidden1_dim * sizeof(float));
    ptr += h.input_dim * h.hidden1_dim;
    memcpy(h.fc1_bias, ptr, h.hidden1_dim * sizeof(float));
    ptr += h.hidden1_dim;

    // LN
    memcpy(h.ln_weight, ptr, h.hidden1_dim * sizeof(float));
    ptr += h.hidden1_dim;
    memcpy(h.ln_bias, ptr, h.hidden1_dim * sizeof(float));
    ptr += h.hidden1_dim;

    // FC2
    memcpy(h.fc2_weight, ptr, h.hidden1_dim * h.hidden2_dim * sizeof(float));
    ptr += h.hidden1_dim * h.hidden2_dim;
    memcpy(h.fc2_bias, ptr, h.hidden2_dim * sizeof(float));
    ptr += h.hidden2_dim;

    // FC3
    memcpy(h.fc3_weight, ptr, h.hidden2_dim * h.output_dim * sizeof(float));
    ptr += h.hidden2_dim * h.output_dim;
    memcpy(h.fc3_bias, ptr, h.output_dim * sizeof(float));

    // Zero momentum after restore (fresh start)
    memset(h.ln_weight_vel, 0, h.hidden1_dim * sizeof(float));
    memset(h.ln_bias_vel, 0, h.hidden1_dim * sizeof(float));
    memset(h.fc2_weight_vel, 0, h.hidden1_dim * h.hidden2_dim * sizeof(float));
    memset(h.fc2_bias_vel, 0, h.hidden2_dim * sizeof(float));
    memset(h.fc3_weight_vel, 0, h.hidden2_dim * h.output_dim * sizeof(float));
    memset(h.fc3_bias_vel, 0, h.output_dim * sizeof(float));

    return true;
}

void GradientEngine::resetToOriginal(uint8_t task_id) {
    if (!_initialized || task_id > 2) return;
    TaskHeadWeights& h = _heads[task_id];

    // Flash weight arrays indexed by task_id
    const float* fc1_w_flash[] = {activity_head_fc1_weight, stress_head_fc1_weight, arrhythmia_head_fc1_weight};
    const float* fc1_b_flash[] = {activity_head_fc1_bias, stress_head_fc1_bias, arrhythmia_head_fc1_bias};
    const float* ln_w_flash[]  = {activity_head_ln_weight, stress_head_ln_weight, arrhythmia_head_ln_weight};
    const float* ln_b_flash[]  = {activity_head_ln_bias, stress_head_ln_bias, arrhythmia_head_ln_bias};
    const float* fc2_w_flash[] = {activity_head_fc2_weight, stress_head_fc2_weight, arrhythmia_head_fc2_weight};
    const float* fc2_b_flash[] = {activity_head_fc2_bias, stress_head_fc2_bias, arrhythmia_head_fc2_bias};
    const float* fc3_w_flash[] = {activity_head_fc3_weight, stress_head_fc3_weight, arrhythmia_head_fc3_weight};
    const float* fc3_b_flash[] = {activity_head_fc3_bias, stress_head_fc3_bias, arrhythmia_head_fc3_bias};

    // Restore all weights from flash
    memcpy(h.fc1_weight, fc1_w_flash[task_id], h.input_dim * h.hidden1_dim * sizeof(float));
    memcpy(h.fc1_bias, fc1_b_flash[task_id], h.hidden1_dim * sizeof(float));
    memcpy(h.ln_weight, ln_w_flash[task_id], h.hidden1_dim * sizeof(float));
    memcpy(h.ln_bias, ln_b_flash[task_id], h.hidden1_dim * sizeof(float));
    memcpy(h.fc2_weight, fc2_w_flash[task_id], h.hidden1_dim * h.hidden2_dim * sizeof(float));
    memcpy(h.fc2_bias, fc2_b_flash[task_id], h.hidden2_dim * sizeof(float));
    memcpy(h.fc3_weight, fc3_w_flash[task_id], h.hidden2_dim * h.output_dim * sizeof(float));
    memcpy(h.fc3_bias, fc3_b_flash[task_id], h.output_dim * sizeof(float));

    // Zero momentum for trainable layers
    memset(h.ln_weight_vel, 0, h.hidden1_dim * sizeof(float));
    memset(h.ln_bias_vel, 0, h.hidden1_dim * sizeof(float));
    memset(h.fc2_weight_vel, 0, h.hidden1_dim * h.hidden2_dim * sizeof(float));
    memset(h.fc2_bias_vel, 0, h.hidden2_dim * sizeof(float));
    memset(h.fc3_weight_vel, 0, h.hidden2_dim * h.output_dim * sizeof(float));
    memset(h.fc3_bias_vel, 0, h.output_dim * sizeof(float));

    CONSOLE.printf("[GradientEngine] Task %d weights reset to original\n", task_id);
}

void GradientEngine::resetAllToOriginal() {
    for (int i = 0; i < 3; i++) {
        resetToOriginal(i);
    }
    _generation = 0;
    _adapted = false;
}

#pragma once

/**
 * On-Device Gradient Engine for ESP32-S3 — V5 Architecture
 *
 * Patent-relevant component: Implements backpropagation through the
 * 3-layer classification head on resource-constrained edge hardware
 * WITHOUT external ML frameworks or GPU.
 *
 * Key innovations (patent claims):
 *   1. Selective backpropagation: only LN+FC2+FC3 parameters are updated,
 *      CNN backbone and FC1 remain frozen — reducing compute by ~95%.
 *   2. Cross-entropy loss with numerical stability for embedded float32.
 *   3. SGD with momentum using pre-allocated, fixed-size buffers.
 *   4. Gradient clipping (L2 norm) to prevent divergence on noisy
 *      real-world biomedical signals.
 *   5. Weight magnitude clamping as a hardware safety constraint.
 *   6. All memory is pre-allocated at init — zero runtime allocations
 *      during training to guarantee deterministic latency.
 *
 * V5 Architecture context:
 *   The inference engine produces a 64-dimensional feature vector from
 *   the frozen CNN+SE+Transformer backbone. This vector is the input
 *   to each 3-layer task head:
 *     - Activity:   [64] -> FC1[64] -> LN[64] -> GELU -> FC2[32] -> GELU -> FC3[4]  -> Softmax
 *     - Stress:     [64] -> FC1[48] -> LN[48] -> GELU -> FC2[24] -> GELU -> FC3[2]  -> Softmax
 *     - Arrhythmia: [64] -> FC1[48] -> LN[48] -> GELU -> FC2[24] -> GELU -> FC3[2]  -> Softmax
 *
 *   FC1 is FROZEN. Backprop computes gradients of cross-entropy loss
 *   w.r.t. LN, FC2, and FC3 weights/biases, then applies SGD with momentum.
 *   Trainable params: 4,984 total (LN+FC2+FC3 across all 3 heads).
 */

#include <Arduino.h>
#include <cmath>
#include "training_config.h"
#include "../inference/nn_inference.h"

// ============================================================
// Training Result Structure
// ============================================================

struct TrainingResult {
    float loss;              // Cross-entropy loss for this step
    float gradient_norm;     // L2 norm of gradients before clipping
    float duration_ms;       // Wall-clock time for this training step
    uint32_t params_updated; // Number of parameters updated
    bool success;            // False if NaN/Inf detected or resource limit hit
};

// ============================================================
// Per-Task Mutable Weight Block (V5 3-layer heads)
// ============================================================
// Holds a copy of the task head weights in PSRAM that can be modified.
// FC1 weights are copied from flash but FROZEN (never updated).
// LN, FC2, FC3 weights are trainable.

struct TaskHeadWeights {
    // Weight arrays
    float* fc1_weight;     // [hidden1_dim, input_dim]  — FROZEN
    float* fc1_bias;       // [hidden1_dim]             — FROZEN
    float* ln_weight;      // [hidden1_dim]             — trainable
    float* ln_bias;        // [hidden1_dim]             — trainable
    float* fc2_weight;     // [hidden2_dim, hidden1_dim] — trainable
    float* fc2_bias;       // [hidden2_dim]             — trainable
    float* fc3_weight;     // [output_dim, hidden2_dim] — trainable
    float* fc3_bias;       // [output_dim]              — trainable

    // Dimensions
    int input_dim;         // FC1 input (64 for all heads)
    int hidden1_dim;       // FC1 output = LN dim
    int hidden2_dim;       // FC2 output
    int output_dim;        // FC3 output (num classes)

    // Momentum buffers for trainable layers only (LN + FC2 + FC3)
    float* ln_weight_vel;
    float* ln_bias_vel;
    float* fc2_weight_vel;
    float* fc2_bias_vel;
    float* fc3_weight_vel;
    float* fc3_bias_vel;

    // Gradient buffers for trainable layers only (LN + FC2 + FC3)
    float* ln_weight_grad;
    float* ln_bias_grad;
    float* fc2_weight_grad;
    float* fc2_bias_grad;
    float* fc3_weight_grad;
    float* fc3_bias_grad;

    // Intermediate activations cached during forward pass (needed for backprop)
    float* fc1_output;     // [hidden1_dim] — pre-LN (raw FC1 output)
    float* ln_output;      // [hidden1_dim] — x_hat (normalized, before scale+shift)
    float* fc1_activated;  // [hidden1_dim] — post-GELU (after LN + GELU)
    float* fc2_output;     // [hidden2_dim] — pre-GELU (raw FC2 output)
    float* fc2_activated;  // [hidden2_dim] — post-GELU
    float* fc3_output;     // [output_dim]  — logits (pre-softmax)

    // LayerNorm cached values for backprop
    float ln_mean;         // mean of fc1_output
    float ln_inv_std;      // 1/sqrt(var + eps)

    size_t totalBytes() const {
        // FC1 weights (frozen copy)
        size_t fc1 = (input_dim * hidden1_dim + hidden1_dim) * sizeof(float);
        // LN weights + momentum + gradients
        size_t ln = (hidden1_dim + hidden1_dim) * sizeof(float) * 3;  // w,b * (weight+vel+grad)
        // FC2 weights + momentum + gradients
        size_t fc2 = (hidden1_dim * hidden2_dim + hidden2_dim) * sizeof(float) * 3;
        // FC3 weights + momentum + gradients
        size_t fc3 = (hidden2_dim * output_dim + output_dim) * sizeof(float) * 3;
        // Activation caches: fc1_output, ln_output, fc1_activated, fc2_output, fc2_activated, fc3_output
        size_t act = (hidden1_dim + hidden1_dim + hidden1_dim +
                      hidden2_dim + hidden2_dim + output_dim) * sizeof(float);
        return fc1 + ln + fc2 + fc3 + act;
    }
};

// ============================================================
// Gradient Engine Class
// ============================================================

class GradientEngine {
public:
    GradientEngine();
    ~GradientEngine();

    /**
     * Initialize the gradient engine.
     * Allocates mutable weight copies in PSRAM and zero-initializes
     * momentum/gradient buffers.
     * @return true if all allocations succeeded
     */
    bool begin();

    /**
     * Run one training step for a specific task head.
     *
     * @param task_id      0=activity, 1=stress, 2=arrhythmia
     * @param features     Pointer to 64-dim feature vector from frozen CNN
     * @param true_label   Ground truth class index
     * @param learning_rate Current learning rate
     * @return TrainingResult with loss, gradient norm, timing
     */
    TrainingResult trainStep(uint8_t task_id,
                             const float* features,
                             uint8_t true_label,
                             float learning_rate);

    /**
     * Run inference using the MUTABLE (adapted) weights instead of
     * the original flash weights.
     *
     * @param task_id  0=activity, 1=stress, 2=arrhythmia
     * @param features 64-dim feature vector
     * @param output   Output probability array (caller-allocated)
     */
    void inferWithAdapted(uint8_t task_id,
                          const float* features,
                          float* output);

    /**
     * Copy current mutable weights to a candidate snapshot buffer.
     * Used by the model store for versioning.
     */
    bool snapshotWeights(uint8_t task_id, float* dest, size_t max_bytes) const;

    /**
     * Restore weights from a snapshot buffer (e.g., rollback).
     */
    bool restoreWeights(uint8_t task_id, const float* src, size_t num_bytes);

    /**
     * Reset adapted weights back to original flash weights.
     */
    void resetToOriginal(uint8_t task_id);
    void resetAllToOriginal();

    /**
     * Get pointer to mutable weights for a task (for external use).
     */
    const TaskHeadWeights& getTaskWeights(uint8_t task_id) const {
        return _heads[task_id];
    }

    /**
     * Get total trainable parameter count.
     */
    uint32_t trainableParamCount() const { return _total_trainable_params; }

    /**
     * Check if engine is initialized.
     */
    bool isReady() const { return _initialized; }

    /**
     * Get the current adaptation generation (incremented per training episode).
     */
    uint32_t generation() const { return _generation; }
    void incrementGeneration() { _generation++; }

    /**
     * Check if any task head has been adapted (weights differ from flash).
     * Used by inference engine to decide whether to use adapted weights.
     */
    bool hasAdaptedWeights() const { return _adapted; }

    /**
     * Mark that weights have been adapted (called after successful training).
     */
    void markAdapted() { _adapted = true; }

    /**
     * Clear adapted flag (called on factory reset).
     */
    void clearAdapted() { _adapted = false; }

private:
    bool _initialized;
    bool _adapted;  // true after any successful training step
    uint32_t _total_trainable_params;
    uint32_t _generation;

    // Three task heads
    TaskHeadWeights _heads[3];

    // ---- Internal helpers ----

    /** Allocate and initialize a single task head's buffers. */
    bool allocTaskHead(TaskHeadWeights& head,
                       int input_dim, int hidden1_dim, int hidden2_dim, int output_dim,
                       const float* flash_fc1_w, const float* flash_fc1_b,
                       const float* flash_ln_w,  const float* flash_ln_b,
                       const float* flash_fc2_w, const float* flash_fc2_b,
                       const float* flash_fc3_w, const float* flash_fc3_b);

    /** Free a task head's buffers. */
    void freeTaskHead(TaskHeadWeights& head);

    /** Forward pass through a 3-layer head, caching intermediates. */
    void headForward(TaskHeadWeights& head, const float* input);

    /** Backward pass: compute gradients of cross-entropy loss. */
    float headBackward(TaskHeadWeights& head, const float* input,
                       uint8_t true_label);

    /** Apply SGD with momentum and gradient clipping. */
    void applyGradients(TaskHeadWeights& head, float lr);

    /** Clip gradient tensors by global L2 norm. */
    float clipGradients(TaskHeadWeights& head);

    /** Clamp all weights to [-MAX, +MAX]. */
    void clampWeights(TaskHeadWeights& head);

    /** Check for NaN/Inf in a buffer. */
    bool hasNanInf(const float* data, int size) const;
};

// Global instance
extern GradientEngine gradientEngine;

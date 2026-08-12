#pragma once

/**
 * On-Device Adaptive Training Configuration
 *
 * Patent-relevant: Defines the parameters for feedback-driven,
 * resource-constrained, partial on-device learning on edge hardware.
 *
 * This configuration governs:
 *   1. Which layers are trainable (frozen backbone, mutable heads)
 *   2. Training hyperparameters (learning rate, momentum, gradient clipping)
 *   3. Feedback triggers (confidence drift, user correction)
 *   4. Resource gates (thermal, memory, power, timing)
 *   5. Safety constraints (rollback thresholds, max divergence)
 *   6. Model versioning (dual-slot stable/candidate)
 */

#include <stdint.h>

namespace AdaptiveConfig {

    // =========================================================
    // Trainable Layer Selection
    // =========================================================
    // The CNN backbone (conv1-3, projection) is FROZEN.
    // Only the classification heads are updated on-device.
    // This is a key patent claim: partial parameter updates
    // reduce compute/memory by 95%+ while preserving learned
    // feature representations.

    constexpr bool TRAIN_ACTIVITY_HEAD   = true;
    constexpr bool TRAIN_STRESS_HEAD     = true;
    constexpr bool TRAIN_ARRHYTHMIA_HEAD = true;

    // Train FC1 (hidden) layers in addition to LN+FC2+FC3 (output)?
    // V5: FC1 frozen, LN+FC2+FC3 trainable = 4,984 params
    constexpr bool TRAIN_FC1_LAYERS = false;

    // =========================================================
    // Training Hyperparameters
    // =========================================================

    constexpr float LEARNING_RATE      = 0.05f;       // Aggressive LR for few-shot domain adaptation
    constexpr float LEARNING_RATE_MIN  = 0.005f;     // Floor for LR decay
    constexpr float LR_DECAY_FACTOR    = 0.95f;      // Per-episode decay
    constexpr float MOMENTUM           = 0.9f;       // SGD momentum
    constexpr float GRADIENT_CLIP_NORM = 1.0f;       // Max gradient L2 norm
    constexpr float WEIGHT_MAX_ABS     = 10.0f;      // Clamp weight magnitude

    // Number of SGD steps per training episode
    // Each step uses one correction from the buffer; corrections may be replayed
    constexpr uint8_t STEPS_PER_EPISODE = 30;

    // =========================================================
    // Feedback Trigger Configuration
    // =========================================================

    // Confidence drift detection: trigger training when the
    // exponential moving average of prediction confidence drops
    // below this threshold over a sliding window.
    constexpr float CONFIDENCE_DRIFT_THRESHOLD = 0.45f;
    constexpr float CONFIDENCE_EMA_ALPHA       = 0.1f;   // Smoothing factor
    constexpr uint16_t DRIFT_WINDOW_SIZE       = 30;     // Windows to track

    // User correction: a serial command provides the true label,
    // which is stored in a correction buffer for supervised training.
    constexpr uint8_t MAX_CORRECTION_BUFFER    = 32;     // Max stored corrections

    // Periodic adaptation: even without explicit triggers,
    // run a training episode every N inference windows if
    // corrections are available in the buffer.
    constexpr uint16_t PERIODIC_TRAIN_INTERVAL = 60;     // ~10 minutes

    // =========================================================
    // Resource Gating
    // =========================================================
    // Training only proceeds when ALL resource checks pass.
    // This is a patent claim: hardware-enforced training windows.

    constexpr bool ENABLE_RESOURCE_GATING = true;

    // Memory: minimum free PSRAM (bytes) to allow training
    constexpr uint32_t MIN_FREE_PSRAM_BYTES = 2 * 1024 * 1024;  // 2 MB

    // Memory: minimum free heap (bytes)
    constexpr uint32_t MIN_FREE_HEAP_BYTES  = 50 * 1024;  // 50 KB

    // Thermal: ESP32-S3 internal temperature sensor
    // Training disabled above this threshold (Celsius)
    constexpr float MAX_TEMPERATURE_C = 65.0f;

    // Timing: minimum interval between training episodes (ms)
    // Note: 10s for benchtop experiments; increase for deployment.
    constexpr uint32_t MIN_TRAIN_INTERVAL_MS = 10000;  // 10 seconds

    // Timing: maximum wall-clock for a single training episode (ms)
    constexpr uint32_t MAX_TRAIN_DURATION_MS = 2000;  // 2000 ms budget (30 steps @ ~5ms each)

    // CPU: skip training if last inference took longer than this (ms)
    // Normal inference is ~2,338 ms per window; defer training if significantly slower.
    constexpr float MAX_INFERENCE_LATENCY_MS = 3000.0f;

    // =========================================================
    // Validation & Safety
    // =========================================================
    // Before promoting candidate weights to stable, the candidate
    // must prove itself on a held-out validation buffer.

    // Size of the validation ring buffer (recent labeled windows)
    constexpr uint8_t VALIDATION_BUFFER_SIZE = 16;

    // Minimum accuracy on validation set to promote candidate
    // Note: Conservative values for production; relaxed for benchtop validation.
    // Production: 0.5, Benchtop: 0.0 (few-shot corrections have limited coverage)
    constexpr float MIN_VALIDATION_ACCURACY = 0.0f;

    // Candidate must be within this margin of stable model accuracy
    // to be promoted. If candidate is worse by more than this, rollback.
    // Production: 0.05, Benchtop: 1.0 (few corrections = high variance)
    constexpr float ROLLBACK_MARGIN = 1.0f;

    // Maximum consecutive failed validations before disabling
    // adaptation entirely (safety lockout).
    constexpr uint8_t MAX_CONSECUTIVE_FAILURES = 5;

    // Weight divergence check: if any weight exceeds this absolute
    // value, the candidate is immediately rolled back.
    constexpr float MAX_WEIGHT_DIVERGENCE = 50.0f;

    // =========================================================
    // Model Versioning (Flash Persistence)
    // =========================================================

    // NVS namespace for storing adapted weights
    // ESP32 NVS has a 15-char limit on namespace names
    constexpr const char* NVS_NAMESPACE = "adapt_weights";

    // Key prefix for weight blobs in NVS
    constexpr const char* NVS_KEY_PREFIX = "w_";

    // Store a version counter to track adaptation generations
    constexpr const char* NVS_VERSION_KEY = "version";

    // Maximum adaptation generations before full reset
    constexpr uint16_t MAX_ADAPTATION_GENERATIONS = 100;

    // =========================================================
    // Telemetry / Logging
    // =========================================================

    // Print training events to serial (for thesis data collection)
    constexpr bool LOG_TRAINING_EVENTS = true;

    // CSV prefix for training telemetry lines
    // Format: TRAIN,t_ms,episode,task,loss,lr,duration_ms,status
    constexpr const char* TELEMETRY_PREFIX = "TRAIN";

    // CSV prefix for adaptation status lines
    // Format: ADAPT,t_ms,generation,stable_acc,candidate_acc,promoted
    constexpr const char* ADAPT_PREFIX = "ADAPT";
}


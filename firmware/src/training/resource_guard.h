#pragma once

/**
 * Resource Guard: Hardware-Enforced Training Windows
 *
 * Patent-relevant component: Implements resource-aware gating that
 * determines whether on-device training is safe to proceed.
 *
 * Key innovations (patent claims):
 *   1. Multi-dimensional resource check: memory, thermal, timing,
 *      and CPU load are all evaluated before allowing training.
 *   2. Hardware-enforced training budget: a wall-clock timeout
 *      ensures training never blocks real-time inference.
 *   3. Thermal guard: uses ESP32-S3 internal temperature sensor
 *      to prevent training when the SoC is thermally stressed.
 *   4. Memory guard: checks both PSRAM and heap availability
 *      to prevent OOM during gradient computation.
 *   5. Cooldown timer: enforces minimum interval between training
 *      episodes to prevent runaway adaptation loops.
 *
 * All checks are non-blocking and O(1) — they add negligible
 * latency to the inference pipeline.
 */

#include <Arduino.h>
#include <esp_heap_caps.h>
#include "training_config.h"

// ============================================================
// Resource Check Result
// ============================================================

struct ResourceCheckResult {
    bool allowed;          // True if all checks pass
    bool memory_ok;        // PSRAM + heap above thresholds
    bool thermal_ok;       // Temperature below limit
    bool timing_ok;        // Cooldown elapsed
    bool cpu_ok;           // Inference latency within budget

    float temperature_c;   // Current SoC temperature
    uint32_t free_psram;   // Current free PSRAM bytes
    uint32_t free_heap;    // Current free heap bytes
    float last_infer_ms;   // Last inference latency
    uint32_t ms_since_last_train;  // Milliseconds since last training
};

// ============================================================
// Resource Guard Class
// ============================================================

class ResourceGuard {
public:
    ResourceGuard();

    /**
     * Initialize the resource guard.
     */
    void begin();

    /**
     * Perform all resource checks.
     * Returns a detailed result structure for telemetry.
     *
     * @param last_inference_ms  The most recent inference latency
     * @return ResourceCheckResult with per-check details
     */
    ResourceCheckResult check(float last_inference_ms) const;

    /**
     * Quick boolean check: can we train right now?
     */
    bool canTrain(float last_inference_ms) const {
        return check(last_inference_ms).allowed;
    }

    /**
     * Mark that a training episode just started.
     */
    void markTrainStart() { _last_train_ms = millis(); }

    /**
     * Mark that a training episode completed.
     * @param duration_ms  How long the episode took
     */
    void markTrainEnd(float duration_ms);

    /**
     * Get the SoC temperature (Celsius).
     * ESP32-S3 has an internal temperature sensor.
     */
    float readTemperature() const;

    /**
     * Get summary string for telemetry.
     */
    void printStatus() const;

    /**
     * Get total training episodes completed.
     */
    uint32_t totalEpisodes() const { return _total_episodes; }

    /**
     * Get average training duration (ms).
     */
    float avgTrainDuration() const {
        return (_total_episodes > 0) ?
            _total_train_ms / _total_episodes : 0.0f;
    }

private:
    uint32_t _last_train_ms;
    uint32_t _total_episodes;
    float _total_train_ms;
};

// Global instance
extern ResourceGuard resourceGuard;

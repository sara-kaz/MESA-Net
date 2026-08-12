#pragma once

/**
 * Validation Engine: Candidate Model Safety Verification
 *
 * Patent-relevant component: Implements a validation pipeline that
 * determines whether adapted model weights are safe to deploy.
 *
 * Key innovations (patent claims):
 *   1. Validation buffer: a ring buffer of recent labeled windows
 *      serves as a hold-out test set for comparing stable vs candidate.
 *   2. A/B comparison: both stable and candidate models are evaluated
 *      on the same validation data, ensuring a fair comparison.
 *   3. Promotion criteria: candidate must achieve accuracy >= stable
 *      accuracy minus a configurable tolerance margin.
 *   4. Consecutive failure tracking: if N consecutive candidates fail,
 *      the system enters a safety lockout and disables adaptation.
 *   5. Weight divergence check: candidate weights are scanned for
 *      extreme values that indicate numerical instability.
 *
 * This ensures that on-device adaptation NEVER degrades the model
 * compared to its pre-adaptation baseline — a critical requirement
 * for medical/safety applications.
 */

#include <Arduino.h>
#include "training_config.h"
#include "gradient_engine.h"
#include "feedback_controller.h"  // For TriggerReason enum

// ============================================================
// Validation Sample (labeled window)
// ============================================================

struct ValidationSample {
    float features[64];    // NNConfig::FEATURE_DIM
    uint8_t labels[3];     // True labels for [activity, stress, arrhythmia]
    bool has_label[3];     // Which labels are valid
    uint32_t timestamp_ms;
    bool valid;
};

// ============================================================
// Validation Result
// ============================================================

struct ValidationResult {
    float stable_accuracy[3];    // Accuracy of stable model per task
    float candidate_accuracy[3]; // Accuracy of candidate model per task
    bool promoted;               // True if candidate was promoted
    bool weight_divergence;      // True if candidate has extreme weights
    uint8_t samples_used;        // Number of validation samples used
    TriggerReason trigger;       // Why this validation was triggered
};

// TriggerReason is defined in feedback_controller.h

// ============================================================
// Validation Engine Class
// ============================================================

class ValidationEngine {
public:
    ValidationEngine();

    /**
     * Initialize the validation engine.
     */
    void begin();

    /**
     * Add a labeled sample to the validation buffer.
     * Called when user provides a correction — the correction is
     * used for both training AND validation (different samples).
     *
     * @param features  64-dim feature vector
     * @param task_id   Which task the label is for
     * @param label     Ground truth class
     */
    void addValidationSample(const float* features, uint8_t task_id,
                              uint8_t label);

    /**
     * Run validation: compare stable vs candidate on the validation buffer.
     * This is called AFTER a training episode, before promotion.
     *
     * @return ValidationResult with detailed accuracy comparison
     */
    ValidationResult validate();

    /**
     * Check if the system is in safety lockout (too many failures).
     */
    bool isLockedOut() const { return _locked_out; }

    /**
     * Reset lockout (e.g., after factory reset or manual override).
     */
    void resetLockout() {
        _locked_out = false;
        _consecutive_failures = 0;
    }

    /**
     * Check for weight divergence in the candidate model.
     * Scans all mutable weights for extreme values.
     */
    bool checkWeightDivergence() const;

    /**
     * Get number of validation samples available.
     */
    uint8_t validationSampleCount() const { return _sample_count; }

    /**
     * Get consecutive failure count.
     */
    uint8_t consecutiveFailures() const { return _consecutive_failures; }

    /**
     * Reset all state.
     */
    void reset();

private:
    // Validation ring buffer
    ValidationSample _samples[AdaptiveConfig::VALIDATION_BUFFER_SIZE];
    uint8_t _sample_head;
    uint8_t _sample_count;

    // Safety lockout
    bool _locked_out;
    uint8_t _consecutive_failures;

    /**
     * Evaluate a model's accuracy on the validation buffer.
     * @param use_adapted  If true, use adapted weights; if false, use stable
     * @param task_id      Which task to evaluate
     * @return Accuracy (0.0 - 1.0)
     */
    float evaluateAccuracy(bool use_adapted, uint8_t task_id);
};

// Global instance
extern ValidationEngine validationEngine;

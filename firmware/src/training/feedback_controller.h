#pragma once

/**
 * Feedback Controller: Drift Detection + User Correction
 *
 * Patent-relevant component: Defines the feedback signals that
 * trigger on-device adaptation. This is the "when to train" logic.
 *
 * Key innovations (patent claims):
 *   1. Confidence drift detection: an exponential moving average (EMA)
 *      of prediction confidence is tracked per task. When it drops
 *      below a threshold, adaptation is triggered — without requiring
 *      explicit labeled data.
 *   2. User correction interface: a serial command protocol allows a
 *      clinician or user to provide the correct label for the most
 *      recent prediction, creating supervised training samples.
 *   3. Correction buffer: stores recent (features, label) pairs as
 *      a mini-batch for training. Ring buffer prevents unbounded growth.
 *   4. Periodic adaptation: even without triggers, accumulated
 *      corrections are consumed on a schedule.
 *   5. Multi-task awareness: each task head has independent drift
 *      tracking and correction buffers.
 *
 * Feedback loop:
 *   Inference -> Confidence EMA -> Drift detected?
 *                                    |
 *                                    v
 *                     User Correction available?
 *                                    |
 *                                    v
 *                         Trigger Training Episode
 */

#include <Arduino.h>
#include "training_config.h"
#include "../inference/nn_inference.h"

// ============================================================
// Correction Sample: a labeled (features, label) pair
// ============================================================

struct CorrectionSample {
    float features[64];    // NNConfig::FEATURE_DIM = 64
    uint8_t label;
    uint8_t task_id;       // 0=activity, 1=stress, 2=arrhythmia
    uint32_t timestamp_ms;
    bool valid;
};

// ============================================================
// Per-Task Drift Tracker
// ============================================================

struct DriftTracker {
    float confidence_ema;    // Exponential moving average of confidence
    uint16_t window_count;   // Number of windows observed
    bool drift_detected;     // True if EMA below threshold
    uint16_t drift_count;    // Consecutive windows below threshold
};

// ============================================================
// Trigger Reason (for telemetry/logging)
// ============================================================

enum class TriggerReason : uint8_t {
    NONE            = 0,
    CONFIDENCE_DRIFT = 1,  // EMA dropped below threshold
    USER_CORRECTION  = 2,  // User provided label correction
    PERIODIC         = 3,  // Scheduled interval
    MANUAL           = 4   // Explicit serial command
};

// ============================================================
// Feedback Controller Class
// ============================================================

class FeedbackController {
public:
    FeedbackController();

    /**
     * Initialize the feedback controller.
     */
    void begin();

    /**
     * Update drift tracking after an inference result.
     * Call this every inference window.
     *
     * @param result  The inference result from the current window
     */
    void updateDrift(const NNResult& result);

    /**
     * Record a user correction for the most recent prediction.
     *
     * @param task_id     0=activity, 1=stress, 2=arrhythmia
     * @param true_label  The correct class label
     * @param features    64-dim feature vector from the CNN backbone
     *                    (must be captured during the inference that
     *                     produced the prediction being corrected)
     * @return true if stored successfully
     */
    bool addCorrection(uint8_t task_id, uint8_t true_label,
                       const float* features);

    /**
     * Check if a training episode should be triggered.
     * Evaluates all trigger conditions:
     *   1. Confidence drift on any task
     *   2. Correction buffer has samples
     *   3. Periodic interval elapsed
     *
     * @param out_reason  Populated with the trigger reason
     * @return true if training should proceed
     */
    bool shouldTrain(TriggerReason& out_reason);

    /**
     * Get the next correction sample for training.
     * Pops from the correction buffer (FIFO).
     *
     * @param out_sample  Populated with the correction data
     * @return true if a sample was available
     */
    bool popCorrection(CorrectionSample& out_sample);

    /**
     * Get the number of pending corrections.
     */
    uint8_t pendingCorrectionCount() const { return _correction_count; }

    /**
     * Get drift tracker for a task (for telemetry).
     */
    const DriftTracker& getDrift(uint8_t task_id) const {
        return _drift[task_id];
    }

    /**
     * Reset all state (after factory reset).
     */
    void reset();

    /**
     * Get total inference windows observed since last training.
     */
    uint16_t windowsSinceLastTrain() const { return _windows_since_train; }

    /**
     * Mark that a training episode just completed.
     */
    void markTrained() { _windows_since_train = 0; }

    /**
     * Cache the latest feature vector from inference.
     * This must be called by the main loop after each inference,
     * so that user corrections can reference it.
     */
    void cacheFeatures(const float* features);

    /**
     * Get the cached feature vector.
     */
    const float* cachedFeatures() const { return _cached_features; }
    bool hasCachedFeatures() const { return _has_cached_features; }

private:
    // Per-task drift tracking
    DriftTracker _drift[3];

    // Correction ring buffer
    CorrectionSample _corrections[AdaptiveConfig::MAX_CORRECTION_BUFFER];
    uint8_t _correction_head;   // Next write position
    uint8_t _correction_tail;   // Next read position
    uint8_t _correction_count;  // Current count

    // Periodic training counter
    uint16_t _windows_since_train;

    // Cached feature vector (from last inference)
    float _cached_features[64];  // NNConfig::FEATURE_DIM
    bool _has_cached_features;
};

// Global instance
extern FeedbackController feedbackController;

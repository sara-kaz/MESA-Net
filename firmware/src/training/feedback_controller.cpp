/**
 * Feedback Controller Implementation
 *
 * Drift detection, user correction buffer, and training trigger logic.
 */

#include "feedback_controller.h"
#include "transport/console.h"

// Global instance
FeedbackController feedbackController;

// ============================================================
// Constructor / Init
// ============================================================

FeedbackController::FeedbackController()
    : _correction_head(0),
      _correction_tail(0),
      _correction_count(0),
      _windows_since_train(0),
      _has_cached_features(false) {
    reset();
}

void FeedbackController::begin() {
    reset();
    CONSOLE.println("[FeedbackController] Feedback controller initialized.");
}

void FeedbackController::reset() {
    for (int i = 0; i < 3; i++) {
        _drift[i].confidence_ema = 1.0f;  // Start optimistic
        _drift[i].window_count = 0;
        _drift[i].drift_detected = false;
        _drift[i].drift_count = 0;
    }
    _correction_head = 0;
    _correction_tail = 0;
    _correction_count = 0;
    _windows_since_train = 0;
    _has_cached_features = false;
    memset(_cached_features, 0, sizeof(_cached_features));
}

// ============================================================
// Drift Detection
// ============================================================

void FeedbackController::updateDrift(const NNResult& result) {
    if (!result.valid) return;

    const float alpha = AdaptiveConfig::CONFIDENCE_EMA_ALPHA;
    const float threshold = AdaptiveConfig::CONFIDENCE_DRIFT_THRESHOLD;

    // Task 0: Activity
    {
        DriftTracker& d = _drift[0];
        d.confidence_ema = alpha * result.activity_confidence +
                           (1.0f - alpha) * d.confidence_ema;
        d.window_count++;
        if (d.confidence_ema < threshold && d.window_count > 10) {
            d.drift_detected = true;
            d.drift_count++;
        } else {
            d.drift_detected = false;
            d.drift_count = 0;
        }
    }

    // Task 1: Stress
    {
        DriftTracker& d = _drift[1];
        d.confidence_ema = alpha * result.stress_confidence +
                           (1.0f - alpha) * d.confidence_ema;
        d.window_count++;
        if (d.confidence_ema < threshold && d.window_count > 10) {
            d.drift_detected = true;
            d.drift_count++;
        } else {
            d.drift_detected = false;
            d.drift_count = 0;
        }
    }

    // Task 2: Arrhythmia
    {
        DriftTracker& d = _drift[2];
        d.confidence_ema = alpha * result.arrhythmia_confidence +
                           (1.0f - alpha) * d.confidence_ema;
        d.window_count++;
        if (d.confidence_ema < threshold && d.window_count > 10) {
            d.drift_detected = true;
            d.drift_count++;
        } else {
            d.drift_detected = false;
            d.drift_count = 0;
        }
    }

    _windows_since_train++;
}

// ============================================================
// User Corrections
// ============================================================

bool FeedbackController::addCorrection(uint8_t task_id, uint8_t true_label,
                                        const float* features) {
    if (task_id > 2 || !features) return false;

    CorrectionSample& sample = _corrections[_correction_head];
    memcpy(sample.features, features, 64 * sizeof(float));
    sample.label = true_label;
    sample.task_id = task_id;
    sample.timestamp_ms = millis();
    sample.valid = true;

    _correction_head = (_correction_head + 1) % AdaptiveConfig::MAX_CORRECTION_BUFFER;

    if (_correction_count < AdaptiveConfig::MAX_CORRECTION_BUFFER) {
        _correction_count++;
    } else {
        // Ring buffer full: advance tail (oldest sample dropped)
        _correction_tail = (_correction_tail + 1) % AdaptiveConfig::MAX_CORRECTION_BUFFER;
    }

    if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
        CONSOLE.printf("[FeedbackController] Correction stored: task=%d label=%d "
                      "buffer=%d/%d\n",
                      task_id, true_label,
                      _correction_count, AdaptiveConfig::MAX_CORRECTION_BUFFER);
    }

    return true;
}

bool FeedbackController::popCorrection(CorrectionSample& out_sample) {
    if (_correction_count == 0) return false;

    out_sample = _corrections[_correction_tail];
    _corrections[_correction_tail].valid = false;
    _correction_tail = (_correction_tail + 1) % AdaptiveConfig::MAX_CORRECTION_BUFFER;
    _correction_count--;

    return true;
}

// ============================================================
// Training Trigger Logic
// ============================================================

bool FeedbackController::shouldTrain(TriggerReason& out_reason) {
    out_reason = TriggerReason::NONE;

    // Priority 1: User corrections available
    if (_correction_count > 0) {
        out_reason = TriggerReason::USER_CORRECTION;
        return true;
    }

    // Priority 2: Confidence drift detected on any task
    for (int t = 0; t < 3; t++) {
        if (_drift[t].drift_detected && _drift[t].drift_count >= 3) {
            out_reason = TriggerReason::CONFIDENCE_DRIFT;
            return true;
        }
    }

    // Priority 3: Periodic interval elapsed (only if corrections buffered)
    // Note: periodic training without corrections would be unsupervised,
    // which we don't support yet. This gate ensures we only train when
    // we have labeled data.
    if (_windows_since_train >= AdaptiveConfig::PERIODIC_TRAIN_INTERVAL &&
        _correction_count > 0) {
        out_reason = TriggerReason::PERIODIC;
        return true;
    }

    return false;
}

// ============================================================
// Feature Caching
// ============================================================

void FeedbackController::cacheFeatures(const float* features) {
    if (features) {
        memcpy(_cached_features, features, 64 * sizeof(float));
        _has_cached_features = true;
    }
}

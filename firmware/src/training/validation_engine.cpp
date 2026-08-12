/**
 * Validation Engine Implementation
 *
 * Candidate model safety verification via A/B accuracy comparison.
 */

#include "validation_engine.h"
#include "transport/console.h"
#include "model_store.h"
#include "transport/console.h"
#include "../inference/model_weights.h"
#include "transport/console.h"

// Global instance
ValidationEngine validationEngine;

// ============================================================
// Constructor / Init
// ============================================================

ValidationEngine::ValidationEngine()
    : _sample_head(0),
      _sample_count(0),
      _locked_out(false),
      _consecutive_failures(0) {
}

void ValidationEngine::begin() {
    reset();
    CONSOLE.println("[ValidationEngine] Validation engine initialized.");
}

void ValidationEngine::reset() {
    _sample_head = 0;
    _sample_count = 0;
    _locked_out = false;
    _consecutive_failures = 0;
    for (int i = 0; i < AdaptiveConfig::VALIDATION_BUFFER_SIZE; i++) {
        _samples[i].valid = false;
    }
}

// ============================================================
// Validation Sample Management
// ============================================================

void ValidationEngine::addValidationSample(const float* features,
                                             uint8_t task_id,
                                             uint8_t label) {
    if (task_id > 2 || !features) return;

    // Find if we already have a sample at this position that could be
    // augmented with labels for other tasks
    // If not, create a new entry
    ValidationSample& s = _samples[_sample_head];

    // If this slot is empty or old, create fresh
    if (!s.valid || (millis() - s.timestamp_ms > 60000)) {
        memcpy(s.features, features, 64 * sizeof(float));
        memset(s.has_label, 0, sizeof(s.has_label));
        memset(s.labels, 0, sizeof(s.labels));
        s.timestamp_ms = millis();
        s.valid = true;
    }

    s.labels[task_id] = label;
    s.has_label[task_id] = true;

    _sample_head = (_sample_head + 1) % AdaptiveConfig::VALIDATION_BUFFER_SIZE;
    if (_sample_count < AdaptiveConfig::VALIDATION_BUFFER_SIZE) {
        _sample_count++;
    }
}

// ============================================================
// Accuracy Evaluation
// ============================================================

float ValidationEngine::evaluateAccuracy(bool /*use_adapted*/, uint8_t task_id) {
    // Evaluates accuracy using whatever weights are currently loaded
    // in the gradient engine. The caller is responsible for loading
    // the appropriate weights (candidate or stable) before calling.
    if (task_id > 2 || _sample_count == 0) return 0.0f;

    int correct = 0;
    int total = 0;

    for (int i = 0; i < AdaptiveConfig::VALIDATION_BUFFER_SIZE; i++) {
        const ValidationSample& s = _samples[i];
        if (!s.valid || !s.has_label[task_id]) continue;

        // Get output dimensions for this task
        int out_dim;
        switch (task_id) {
            case 0: out_dim = NNConfig::ACTIVITY_CLASSES; break;
            case 1: out_dim = NNConfig::STRESS_CLASSES; break;
            case 2: out_dim = NNConfig::ARRHYTHMIA_CLASSES; break;
            default: continue;
        }

        float probs[16];  // Max 8 classes
        gradientEngine.inferWithAdapted(task_id, s.features, probs);

        // Argmax
        uint8_t pred = 0;
        float max_prob = probs[0];
        for (int j = 1; j < out_dim; j++) {
            if (probs[j] > max_prob) {
                max_prob = probs[j];
                pred = j;
            }
        }

        if (pred == s.labels[task_id]) correct++;
        total++;
    }

    return (total > 0) ? (float)correct / (float)total : 0.0f;
}

// ============================================================
// Weight Divergence Check
// ============================================================

bool ValidationEngine::checkWeightDivergence() const {
    const float max_div = AdaptiveConfig::MAX_WEIGHT_DIVERGENCE;

    for (uint8_t t = 0; t < 3; t++) {
        const TaskHeadWeights& h = gradientEngine.getTaskWeights(t);

        // Check LN weights (trainable)
        for (int i = 0; i < h.hidden1_dim; i++) {
            if (fabsf(h.ln_weight[i]) > max_div) return true;
            if (fabsf(h.ln_bias[i]) > max_div) return true;
        }

        // Check FC2 weights (trainable)
        for (int i = 0; i < h.hidden1_dim * h.hidden2_dim; i++) {
            if (fabsf(h.fc2_weight[i]) > max_div) return true;
        }
        for (int i = 0; i < h.hidden2_dim; i++) {
            if (fabsf(h.fc2_bias[i]) > max_div) return true;
        }

        // Check FC3 weights (trainable)
        for (int i = 0; i < h.hidden2_dim * h.output_dim; i++) {
            if (fabsf(h.fc3_weight[i]) > max_div) return true;
        }
        for (int i = 0; i < h.output_dim; i++) {
            if (fabsf(h.fc3_bias[i]) > max_div) return true;
        }
    }

    return false;
}

// ============================================================
// Full Validation Pipeline
// ============================================================

ValidationResult ValidationEngine::validate() {
    ValidationResult result = {};
    result.promoted = false;
    result.weight_divergence = false;
    result.samples_used = _sample_count;

    // Insufficient validation data — skip validation, don't promote
    if (_sample_count < 2) {
        if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
            CONSOLE.printf("[ValidationEngine] Insufficient samples (%d), "
                          "skipping validation\n", _sample_count);
        }
        return result;
    }

    // 1. Check weight divergence first (fast check)
    if (checkWeightDivergence()) {
        CONSOLE.println("[ValidationEngine] WEIGHT DIVERGENCE detected — "
                       "candidate rejected");
        result.weight_divergence = true;
        _consecutive_failures++;

        if (_consecutive_failures >= AdaptiveConfig::MAX_CONSECUTIVE_FAILURES) {
            _locked_out = true;
            CONSOLE.println("[ValidationEngine] SAFETY LOCKOUT: too many "
                           "consecutive failures");
        }
        return result;
    }

    // 2. Snapshot candidate weights into the model store's candidate slot
    //    so we can restore them after evaluating stable.
    modelStore.snapshotCandidate();

    // 3. Evaluate candidate accuracy (current adapted weights are loaded)
    for (int t = 0; t < 3; t++) {
        result.candidate_accuracy[t] = evaluateAccuracy(true, t);
    }

    // 4. Evaluate stable accuracy:
    //    - Rollback loads stable weights from model store into gradient engine
    //    - Evaluate accuracy with those stable weights
    //    - Then restore candidate weights from model store's candidate slot
    modelStore.rollback();  // Loads stable weights into gradient engine
    for (int t = 0; t < 3; t++) {
        result.stable_accuracy[t] = evaluateAccuracy(true, t);
    }

    // 5. Restore candidate weights back into the gradient engine
    //    (candidate slot was saved in step 2; rollback only changed the
    //     gradient engine's live weights, not the candidate slot buffer)
    modelStore.restoreCandidate();

    // 5. Promotion decision
    bool should_promote = true;
    for (int t = 0; t < 3; t++) {
        // Candidate must be within ROLLBACK_MARGIN of stable
        if (result.candidate_accuracy[t] <
            result.stable_accuracy[t] - AdaptiveConfig::ROLLBACK_MARGIN) {
            should_promote = false;
            break;
        }
        // Candidate must meet minimum absolute accuracy
        if (result.candidate_accuracy[t] < AdaptiveConfig::MIN_VALIDATION_ACCURACY) {
            should_promote = false;
            break;
        }
    }

    if (should_promote) {
        result.promoted = true;
        _consecutive_failures = 0;

        if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
            CONSOLE.printf("[ValidationEngine] Candidate PROMOTED "
                          "(act=%.1f%% str=%.1f%% arr=%.1f%%)\n",
                          result.candidate_accuracy[0] * 100.0f,
                          result.candidate_accuracy[1] * 100.0f,
                          result.candidate_accuracy[2] * 100.0f);
        }
    } else {
        _consecutive_failures++;

        if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
            CONSOLE.printf("[ValidationEngine] Candidate REJECTED "
                          "(failures=%d/%d)\n",
                          _consecutive_failures,
                          AdaptiveConfig::MAX_CONSECUTIVE_FAILURES);
        }

        if (_consecutive_failures >= AdaptiveConfig::MAX_CONSECUTIVE_FAILURES) {
            _locked_out = true;
            CONSOLE.println("[ValidationEngine] SAFETY LOCKOUT activated");
        }
    }

    return result;
}

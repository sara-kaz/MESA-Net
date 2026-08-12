#pragma once

/**
 * Adaptive Trainer: On-Device Learning Orchestrator
 *
 * Patent-relevant component: The top-level controller that orchestrates
 * all adaptive learning subsystems into a complete feedback-driven
 * training pipeline on edge hardware.
 *
 * This is the "brain" of the patent — it coordinates:
 *   1. Feedback Controller → "Should we train?"
 *   2. Resource Guard      → "Can we train safely?"
 *   3. Gradient Engine      → "Execute training"
 *   4. Validation Engine    → "Is the result better?"
 *   5. Model Store          → "Persist or rollback"
 *
 * Complete adaptation cycle:
 *
 *   [Inference] → confidence + predictions
 *        |
 *        v
 *   [FeedbackController.updateDrift()]  ← track confidence EMA
 *        |
 *        v
 *   [User Correction]  ← optional: serial command with true label
 *        |
 *        v
 *   [shouldTrain()?]  ← drift / correction / periodic trigger
 *        |
 *        ├── No  → continue inference
 *        |
 *        └── Yes → [ResourceGuard.canTrain()?]
 *                       |
 *                       ├── No  → defer, continue inference
 *                       |
 *                       └── Yes → [runTrainingEpisode()]
 *                                      |
 *                                      v
 *                              Gradient steps (N steps on correction buffer)
 *                                      |
 *                                      v
 *                              [ValidationEngine.validate()]
 *                                      |
 *                                 ┌────┴────┐
 *                                 |         |
 *                              Promote   Rollback
 *                                 |         |
 *                                 v         v
 *                            [Persist]  [Restore stable]
 *                            to NVS      from snapshot
 *
 * Serial command protocol for user interaction:
 *   CORRECT <task> <label>  — provide correction for last prediction
 *   TRAIN                   — force a training episode
 *   RESET                   — factory reset all adapted weights
 *   STATUS                  — print adaptation status
 */

#include <Arduino.h>
#include "training_config.h"
#include "gradient_engine.h"
#include "model_store.h"
#include "feedback_controller.h"
#include "resource_guard.h"
#include "validation_engine.h"

// ============================================================
// Training Episode Result
// ============================================================

struct EpisodeResult {
    uint8_t steps_completed;
    float avg_loss;
    float avg_gradient_norm;
    float total_duration_ms;
    bool validated;
    bool promoted;
    bool rolled_back;
    TriggerReason trigger;
};

// ============================================================
// Adaptive Trainer Class
// ============================================================

class AdaptiveTrainer {
public:
    AdaptiveTrainer();

    /**
     * Initialize all training subsystems.
     * Must be called after nnInference.begin().
     * @return true if all subsystems initialized
     */
    bool begin();

    /**
     * Called after each inference window.
     * Updates drift tracking, checks triggers, and runs training
     * if conditions are met.
     *
     * @param result       Inference result from current window
     * @param features     64-dim CNN features (cached from inference)
     * @param infer_ms     Inference latency for resource check
     */
    void onInferenceComplete(const NNResult& result,
                              const float* features,
                              float infer_ms);

    /**
     * Process a serial command for the adaptive training system.
     * Parses commands like "CORRECT 0 3" or "TRAIN" or "RESET".
     *
     * @param command  Null-terminated command string
     * @return true if the command was handled
     */
    bool handleCommand(const char* command);

    /**
     * Force a training episode (manual trigger).
     */
    EpisodeResult forceTrainEpisode();

    /**
     * Print comprehensive status of all subsystems.
     */
    void printStatus() const;

    /**
     * Check if the system is ready for adaptive training.
     */
    bool isReady() const { return _initialized; }

    /**
     * Check if adaptation is currently enabled (not locked out).
     */
    bool isAdaptationEnabled() const {
        return _initialized && !validationEngine.isLockedOut();
    }

    /**
     * Get total episodes run.
     */
    uint32_t totalEpisodes() const { return _total_episodes; }

    /**
     * Get current learning rate.
     */
    float currentLearningRate() const { return _current_lr; }

private:
    bool _initialized;
    uint32_t _total_episodes;
    float _current_lr;

    /**
     * Execute a full training episode:
     *   1. Pop corrections from buffer
     *   2. Run gradient steps
     *   3. Validate candidate
     *   4. Promote or rollback
     */
    EpisodeResult runEpisode(TriggerReason trigger);

    /**
     * Parse a "CORRECT <task> <label>" command.
     */
    bool parseCorrectCommand(const char* args);
};

// Global instance
extern AdaptiveTrainer adaptiveTrainer;
